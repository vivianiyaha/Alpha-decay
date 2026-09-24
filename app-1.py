"""
=====================================================================================
 MULTI-ASSET QUANTITATIVE SIGNAL BOT
 Forex | Metals (Gold/Silver) | Crypto
---------------------------------------------------------------------------------
 A production-grade Streamlit application implementing:
   - Statistical "Edge" detection (Mean-Reversion / Volatility-Breakout)
   - A Multi-Factor Confidence Score (statistical significance, volume, trend)
   - A strict 75% confidence execution filter
   - ATR-based Entry / Stop-Loss / Take-Profit engine
   - Alpha Decay modeling via return autocorrelation -> Alpha Half-Life
   - A position-sizing Risk Management calculator

 Author: Senior Quantitative Developer / FinTech Architect (generated)
 Run:    streamlit run app.py
=====================================================================================
"""

import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import curve_fit
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

# =====================================================================================
# 1. GLOBAL CONFIGURATION
# =====================================================================================

st.set_page_config(
    page_title="Quant Signal Bot | Forex · Metals · Crypto",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Universe of tradable instruments, grouped by asset class, mapped to yfinance tickers.
ASSET_UNIVERSE: dict = {
    "Forex": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CHF": "USDCHF=X",
    },
    "Metals": {
        "Gold (XAU/USD)": "XAUUSD=X",
        "Silver (XAG/USD)": "XAGUSD=X",
    },
    "Crypto": {
        "BTC/USD": "BTC-USD",
        "ETH/USD": "ETH-USD",
        "SOL/USD": "SOL-USD",
    },
}

# Timeframe -> (yfinance fetch interval, fetch period, optional resample rule)
TIMEFRAME_MAP: dict = {
    "1H": {"interval": "1h", "period": "60d", "resample": None},
    "4H": {"interval": "1h", "period": "60d", "resample": "4h"},
    "1D": {"interval": "1d", "period": "2y", "resample": None},
}

MIN_CONFIDENCE_THRESHOLD = 75.0  # Hard rule: signals below this are NEUTRAL / NO TRADE
CACHE_TTL_SECONDS = 300


# =====================================================================================
# 2. DATA STRUCTURES
# =====================================================================================

@dataclass
class SignalResult:
    """Container for one asset's full quantitative signal output."""
    asset_name: str
    ticker: str
    asset_class: str
    current_price: float = np.nan
    signal: str = "NEUTRAL"                 # BUY / SELL / NEUTRAL
    confidence: float = 0.0                 # 0-100
    z_score: float = 0.0
    p_value: float = 1.0
    volume_alignment: float = 0.0
    trend_strength: float = 0.0
    atr: float = np.nan
    entry: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    alpha_half_life: Optional[float] = None
    acf_values: list = field(default_factory=list)
    df: Optional[pd.DataFrame] = None
    error: Optional[str] = None


# =====================================================================================
# 3. DATA INGESTION & NORMALIZATION MODULE
# =====================================================================================

class DataManager:
    """Handles fetching, cleaning, and normalizing OHLCV data across timeframes."""

    @staticmethod
    @st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
    def fetch_ohlcv(ticker: str, timeframe: str) -> pd.DataFrame:
        cfg = TIMEFRAME_MAP[timeframe]
        try:
            raw = yf.download(
                ticker,
                interval=cfg["interval"],
                period=cfg["period"],
                progress=False,
                auto_adjust=True,
            )
        except Exception:
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        # yfinance can return MultiIndex columns for some tickers/versions
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        if cfg["resample"]:
            raw = raw.resample(cfg["resample"]).agg(
                {"Open": "first", "High": "max", "Low": "min",
                 "Close": "last", "Volume": "sum"}
            )

        return DataManager.clean(raw)

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values, drop degenerate rows, compute baseline returns."""
        if df.empty:
            return df
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().dropna()
        df = df[(df["Volume"] >= 0) & (df["High"] >= df["Low"])]
        df["returns"] = df["Close"].pct_change()
        df = df.dropna()
        return df


# =====================================================================================
# 4. EDGE & CONFIDENCE CALCULATION ENGINE
# =====================================================================================

class EdgeEngine:
    """
    Implements two quantitative strategies:
      - Mean-Reversion (Z-score of price deviation from rolling mean)
      - Volatility-Breakout (price breakout of rolling range, volume-confirmed)

    Combines statistical significance, volume alignment, and trend strength into
    a single 0-100 Multi-Factor Confidence Score.
    """

    def __init__(self, zscore_window: int = 20, breakout_window: int = 20,
                 atr_window: int = 14, adx_window: int = 14):
        self.zscore_window = zscore_window
        self.breakout_window = breakout_window
        self.atr_window = atr_window
        self.adx_window = adx_window

    # ---------------------------------------------------------------------- indicators
    @staticmethod
    def compute_zscore(series: pd.Series, window: int) -> pd.Series:
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std.replace(0, np.nan)

    def compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(self.atr_window).mean()

    def compute_adx(self, df: pd.DataFrame) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        atr = self.compute_atr(df).replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(self.adx_window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_window).mean() / atr)

        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = ((plus_di - minus_di).abs() / denom) * 100
        adx = dx.rolling(self.adx_window).mean()
        return adx.fillna(0)

    # ---------------------------------------------------------------------- strategies
    def mean_reversion_direction(self, z: float) -> str:
        if z <= -2.0:
            return "BUY"   # price oversold relative to mean -> expect reversion up
        if z >= 2.0:
            return "SELL"  # price overbought relative to mean -> expect reversion down
        return "NEUTRAL"

    def breakout_direction(self, df: pd.DataFrame) -> str:
        window = self.breakout_window
        rolling_high = df["High"].rolling(window).max().shift(1)
        rolling_low = df["Low"].rolling(window).min().shift(1)
        last_close = df["Close"].iloc[-1]
        if last_close > rolling_high.iloc[-1]:
            return "BUY"
        if last_close < rolling_low.iloc[-1]:
            return "SELL"
        return "NEUTRAL"

    # ---------------------------------------------------------------------- confidence
    def evaluate(self, df: pd.DataFrame, strategy: str) -> dict:
        """Runs the selected strategy and returns a dict of raw factor values."""
        z_series = self.compute_zscore(df["Close"], self.zscore_window)
        z = float(z_series.iloc[-1]) if not np.isnan(z_series.iloc[-1]) else 0.0
        p_value = float(2 * stats.norm.sf(abs(z)))  # two-tailed significance

        atr_series = self.compute_atr(df)
        adx_series = self.compute_adx(df)
        adx = float(adx_series.iloc[-1]) if len(adx_series) else 0.0

        avg_volume = df["Volume"].rolling(20).mean().iloc[-1]
        cur_volume = df["Volume"].iloc[-1]
        volume_ratio = (cur_volume / avg_volume) if avg_volume and avg_volume > 0 else 1.0

        if strategy == "Mean-Reversion":
            direction = self.mean_reversion_direction(z)
            stat_significance = float(np.clip((1 - p_value) * 100, 0, 100))
        else:  # Volatility-Breakout
            direction = self.breakout_direction(df)
            # Significance proxied by how far price extends beyond the breakout band,
            # measured in ATR units (larger extension => stronger statistical edge).
            atr_last = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else np.nan
            if atr_last and atr_last > 0:
                extension = abs(df["Close"].iloc[-1] - df["Close"].rolling(
                    self.breakout_window).mean().iloc[-1]) / atr_last
            else:
                extension = 0.0
            stat_significance = float(np.clip(extension * 25, 0, 100))

        # Volume alignment: reward above-average participation, cap contribution at 2x
        volume_alignment = float(np.clip((volume_ratio / 2.0) * 100, 0, 100))

        # Trend strength from ADX (ADX > 25 = trending, > 50 = strong trend)
        trend_strength = float(np.clip(adx * 2, 0, 100))

        confidence = (
            0.40 * stat_significance +
            0.30 * volume_alignment +
            0.30 * trend_strength
        )

        if confidence < MIN_CONFIDENCE_THRESHOLD:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "confidence": round(confidence, 2),
            "z_score": round(z, 3),
            "p_value": round(p_value, 4),
            "volume_alignment": round(volume_alignment, 2),
            "trend_strength": round(trend_strength, 2),
            "atr": float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else 0.0,
        }


# =====================================================================================
# 5. ALPHA DECAY TRACKING ENGINE
# =====================================================================================

class AlphaDecayEngine:
    """
    Models Alpha Decay using the autocorrelation function (ACF) of returns.
    Fits an exponential decay curve |ACF(lag)| = a * exp(-lambda * lag) and
    derives the Alpha Half-Life: the holding period at which predictive edge
    has decayed to 50% of its initial value.
    """

    def __init__(self, max_lag: int = 20):
        self.max_lag = max_lag

    def compute_acf(self, returns: pd.Series) -> list:
        returns = returns.dropna().values
        acf_vals = []
        for lag in range(1, self.max_lag + 1):
            if len(returns) <= lag + 1:
                acf_vals.append(0.0)
                continue
            s1 = returns[:-lag]
            s2 = returns[lag:]
            if np.std(s1) == 0 or np.std(s2) == 0:
                acf_vals.append(0.0)
            else:
                acf_vals.append(float(np.corrcoef(s1, s2)[0, 1]))
        return acf_vals

    @staticmethod
    def _exp_decay(lag, a, lam):
        return a * np.exp(-lam * lag)

    def compute_half_life(self, acf_vals: list) -> float:
        """Fits an exponential decay to |ACF| and solves for the half-life in periods."""
        lags = np.arange(1, len(acf_vals) + 1)
        abs_acf = np.abs(np.array(acf_vals))

        if abs_acf.max() < 1e-6:
            return float(self.max_lag)  # negligible autocorrelation -> treat as slow decay

        try:
            popt, _ = curve_fit(
                self._exp_decay, lags, abs_acf,
                p0=[abs_acf[0] if abs_acf[0] > 0 else 0.1, 0.2],
                maxfev=5000,
                bounds=([0, 1e-4], [1, 5]),
            )
            _, lam = popt
            if lam <= 0:
                return float(self.max_lag)
            half_life = np.log(2) / lam
            return float(np.clip(half_life, 0.5, self.max_lag * 2))
        except Exception:
            # Fallback: first lag where |ACF| drops below half of lag-1 value
            target = abs_acf[0] / 2.0
            below = np.where(abs_acf <= target)[0]
            return float(below[0] + 1) if len(below) else float(self.max_lag)


# =====================================================================================
# 6. TRADE EXECUTION ENGINE (Entry / SL / TP)
# =====================================================================================

class TradeExecutionEngine:
    """Computes precise Entry, Stop-Loss, and Take-Profit levels for a signal
    that has passed the confidence filter, adjusting reward targets for Alpha Decay."""

    def __init__(self, atr_multiplier: float = 1.5, base_rr: float = 2.0):
        self.atr_multiplier = atr_multiplier
        self.base_rr = base_rr

    def compute_levels(self, direction: str, price: float, atr: float,
                        half_life: float) -> dict:
        if direction == "NEUTRAL" or atr <= 0 or np.isnan(atr):
            return {"entry": None, "stop_loss": None, "take_profit": None, "rr": None}

        entry = price

        # Alpha-decay-scaled Risk:Reward -- short half-life demands faster profit-taking
        # (lower RR target), long half-life allows the trade more room to run.
        decay_factor = np.clip(half_life / 10.0, 0.5, 2.0)
        effective_rr = round(float(np.clip(self.base_rr * decay_factor, 1.0, 4.0)), 2)

        if direction == "BUY":
            stop_loss = entry - self.atr_multiplier * atr
            risk = entry - stop_loss
            take_profit = entry + risk * effective_rr
        else:  # SELL
            stop_loss = entry + self.atr_multiplier * atr
            risk = stop_loss - entry
            take_profit = entry - risk * effective_rr

        return {
            "entry": round(float(entry), 5),
            "stop_loss": round(float(stop_loss), 5),
            "take_profit": round(float(take_profit), 5),
            "rr": effective_rr,
        }


# =====================================================================================
# 7. RISK MANAGEMENT / POSITION SIZING CALCULATOR
# =====================================================================================

class RiskManager:
    """Computes position size from account risk tolerance, stop distance, and the
    signal's alpha-decay weight (shorter half-life -> smaller size, all else equal)."""

    @staticmethod
    def position_size(account_balance: float, risk_pct: float,
                       entry: float, stop_loss: float, half_life: float,
                       max_half_life: float = 20.0) -> dict:
        if entry is None or stop_loss is None or entry == stop_loss:
            return {"risk_amount": 0.0, "units": 0.0, "decay_weight": 0.0}

        risk_amount = account_balance * (risk_pct / 100.0)
        stop_distance = abs(entry - stop_loss)

        # Decay weight: fraction of full size warranted given how quickly the edge fades
        decay_weight = float(np.clip(half_life / max_half_life, 0.25, 1.0))

        raw_units = risk_amount / stop_distance
        adjusted_units = raw_units * decay_weight

        return {
            "risk_amount": round(risk_amount, 2),
            "units": round(adjusted_units, 4),
            "decay_weight": round(decay_weight, 2),
        }


# =====================================================================================
# 8. CHART BUILDER (Plotly)
# =====================================================================================

class ChartBuilder:

    @staticmethod
    def price_chart(df: pd.DataFrame, result: SignalResult, asset_label: str) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
            vertical_spacing=0.03,
            subplot_titles=(f"{asset_label} — Price Action", "Volume"),
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Price",
                increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            ),
            row=1, col=1,
        )

        fig.add_trace(
            go.Bar(x=df.index, y=df["Volume"], name="Volume",
                   marker_color="#7e57c2", opacity=0.6),
            row=2, col=1,
        )

        # Overlay Entry / SL / TP as horizontal reference lines
        line_specs = [
            (result.entry, "Entry", "#42a5f5"),
            (result.stop_loss, "Stop Loss", "#ef5350"),
            (result.take_profit, "Take Profit", "#26a69a"),
        ]
        for level, label, color in line_specs:
            if level is not None:
                fig.add_hline(
                    y=level, line_dash="dash", line_color=color,
                    annotation_text=f"{label}: {level}",
                    annotation_position="right",
                    row=1, col=1,
                )

        fig.update_layout(
            height=600, template="plotly_dark",
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=60, b=20, l=20, r=20),
        )
        return fig

    @staticmethod
    def decay_curve(acf_vals: list, half_life: float, asset_label: str) -> go.Figure:
        lags = list(range(1, len(acf_vals) + 1))
        abs_acf = [abs(v) for v in acf_vals]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=lags, y=acf_vals, name="Autocorrelation (signed)",
            marker_color="#42a5f5", opacity=0.7,
        ))
        fig.add_trace(go.Scatter(
            x=lags, y=abs_acf, name="|Autocorrelation|",
            mode="lines+markers", line=dict(color="#ffb300", width=2),
        ))
        fig.add_vline(
            x=half_life, line_dash="dash", line_color="#ef5350",
            annotation_text=f"Alpha Half-Life ≈ {half_life:.1f} periods",
            annotation_position="top",
        )
        fig.update_layout(
            title=f"{asset_label} — Alpha Decay Curve (Return Autocorrelation)",
            xaxis_title="Holding Period Lag", yaxis_title="Autocorrelation",
            height=420, template="plotly_dark",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        return fig

    @staticmethod
    def equity_curve_proxy(df: pd.DataFrame, asset_label: str) -> go.Figure:
        """Illustrative cumulative-return curve for the asset's recent history."""
        cum_returns = (1 + df["returns"]).cumprod() - 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=cum_returns * 100, mode="lines",
            line=dict(color="#26a69a", width=2), name="Cumulative Return (%)",
        ))
        fig.update_layout(
            title=f"{asset_label} — Cumulative Return (Buy & Hold Reference)",
            yaxis_title="Return (%)", height=320, template="plotly_dark",
            margin=dict(t=60, b=20, l=20, r=20),
        )
        return fig


# =====================================================================================
# 9. SIGNAL ORCHESTRATION
# =====================================================================================

def build_signal(asset_name: str, ticker: str, asset_class: str, timeframe: str,
                  strategy: str, edge_engine: EdgeEngine, decay_engine: AlphaDecayEngine,
                  trade_engine: TradeExecutionEngine) -> SignalResult:
    result = SignalResult(asset_name=asset_name, ticker=ticker, asset_class=asset_class)

    df = DataManager.fetch_ohlcv(ticker, timeframe)
    min_bars = max(edge_engine.zscore_window, edge_engine.breakout_window,
                    edge_engine.adx_window) + 10

    if df.empty or len(df) < min_bars:
        result.error = "Insufficient or unavailable market data."
        return result

    factors = edge_engine.evaluate(df, strategy)
    acf_vals = decay_engine.compute_acf(df["returns"])
    half_life = decay_engine.compute_half_life(acf_vals)

    levels = trade_engine.compute_levels(
        direction=factors["direction"],
        price=float(df["Close"].iloc[-1]),
        atr=factors["atr"],
        half_life=half_life,
    )

    result.current_price = round(float(df["Close"].iloc[-1]), 5)
    result.signal = factors["direction"]
    result.confidence = factors["confidence"]
    result.z_score = factors["z_score"]
    result.p_value = factors["p_value"]
    result.volume_alignment = factors["volume_alignment"]
    result.trend_strength = factors["trend_strength"]
    result.atr = round(factors["atr"], 5)
    result.entry = levels["entry"]
    result.stop_loss = levels["stop_loss"]
    result.take_profit = levels["take_profit"]
    result.risk_reward = levels["rr"]
    result.alpha_half_life = round(half_life, 2)
    result.acf_values = acf_vals
    result.df = df
    return result


# =====================================================================================
# 10. STREAMLIT DASHBOARD UI
# =====================================================================================

def render_sidebar():
    st.sidebar.title("⚙️ Bot Configuration")

    st.sidebar.subheader("Asset Selection")
    asset_class = st.sidebar.selectbox("Asset Class", list(ASSET_UNIVERSE.keys()))
    available_assets = list(ASSET_UNIVERSE[asset_class].keys())
    selected_assets = st.sidebar.multiselect(
        "Instruments", available_assets, default=available_assets[:3]
    )

    timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAME_MAP.keys()), index=1)

    st.sidebar.subheader("Strategy")
    strategy = st.sidebar.radio(
        "Edge Model", ["Mean-Reversion", "Volatility-Breakout"]
    )

    st.sidebar.subheader("Strategy Parameters")
    zscore_window = st.sidebar.slider("Z-Score Lookback (bars)", 10, 60, 20)
    breakout_window = st.sidebar.slider("Breakout Lookback (bars)", 10, 60, 20)
    atr_window = st.sidebar.slider("ATR Window", 5, 30, 14)
    atr_multiplier = st.sidebar.slider("SL — ATR Multiplier", 0.5, 4.0, 1.5, 0.1)
    base_rr = st.sidebar.slider("Base Risk:Reward Ratio", 1.0, 4.0, 2.0, 0.5)

    st.sidebar.caption(
        f"🔒 Hard rule: signals fire only at ≥ {MIN_CONFIDENCE_THRESHOLD:.0f}% confidence."
    )

    st.sidebar.subheader("Risk Tolerance")
    account_balance = st.sidebar.number_input(
        "Account Balance (USD)", min_value=100.0, value=10000.0, step=100.0
    )
    risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.25, 5.0, 1.0, 0.25)

    return {
        "asset_class": asset_class,
        "selected_assets": selected_assets,
        "timeframe": timeframe,
        "strategy": strategy,
        "zscore_window": zscore_window,
        "breakout_window": breakout_window,
        "atr_window": atr_window,
        "atr_multiplier": atr_multiplier,
        "base_rr": base_rr,
        "account_balance": account_balance,
        "risk_pct": risk_pct,
    }


def style_signal_table(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def color_signal(val):
        if val == "BUY":
            return "color: #26a69a; font-weight: 700"
        if val == "SELL":
            return "color: #ef5350; font-weight: 700"
        return "color: #9e9e9e"

    def color_confidence(val):
        try:
            v = float(val)
        except (TypeError, ValueError):
            return ""
        if v >= MIN_CONFIDENCE_THRESHOLD:
            return "color: #26a69a; font-weight: 700"
        return "color: #9e9e9e"

    styler = df.style.applymap(color_signal, subset=["Signal"])
    styler = styler.applymap(color_confidence, subset=["Confidence %"])
    return styler


def main():
    st.title("📊 Multi-Asset Quantitative Signal Bot")
    st.caption(
        "Forex · Metals · Crypto — Edge detection, confidence-filtered execution "
        "levels, and Alpha Decay analytics."
    )

    cfg = render_sidebar()

    if not cfg["selected_assets"]:
        st.warning("Select at least one instrument from the sidebar to begin.")
        return

    edge_engine = EdgeEngine(
        zscore_window=cfg["zscore_window"],
        breakout_window=cfg["breakout_window"],
        atr_window=cfg["atr_window"],
        adx_window=cfg["atr_window"],
    )
    decay_engine = AlphaDecayEngine(max_lag=20)
    trade_engine = TradeExecutionEngine(
        atr_multiplier=cfg["atr_multiplier"], base_rr=cfg["base_rr"]
    )

    results: list[SignalResult] = []
    with st.spinner("Fetching market data and computing signals..."):
        for asset_name in cfg["selected_assets"]:
            ticker = ASSET_UNIVERSE[cfg["asset_class"]][asset_name]
            res = build_signal(
                asset_name, ticker, cfg["asset_class"], cfg["timeframe"],
                cfg["strategy"], edge_engine, decay_engine, trade_engine,
            )
            results.append(res)

    valid_results = [r for r in results if r.error is None]
    error_results = [r for r in results if r.error is not None]

    tab_signals, tab_chart, tab_decay, tab_risk = st.tabs(
        ["📋 Signal Table", "📈 Price Chart", "🧬 Alpha Decay Analytics", "💰 Risk Calculator"]
    )

    # ---------------------------------------------------------------- TAB 1: SIGNALS
    with tab_signals:
        st.subheader(f"Real-Time Signals — {cfg['strategy']} · {cfg['timeframe']}")

        if valid_results:
            table_rows = []
            for r in valid_results:
                table_rows.append({
                    "Asset": r.asset_name,
                    "Price": r.current_price,
                    "Signal": r.signal,
                    "Confidence %": r.confidence,
                    "Entry": r.entry if r.entry is not None else "—",
                    "Stop Loss": r.stop_loss if r.stop_loss is not None else "—",
                    "Take Profit": r.take_profit if r.take_profit is not None else "—",
                    "R:R": r.risk_reward if r.risk_reward is not None else "—",
                    "Alpha Half-Life (bars)": r.alpha_half_life,
                })
            table_df = pd.DataFrame(table_rows)
            st.dataframe(style_signal_table(table_df), use_container_width=True, height=38 * len(table_df) + 40)

            actionable = [r for r in valid_results if r.signal != "NEUTRAL"]
            st.metric("Actionable Signals (≥75% confidence)", len(actionable))
        else:
            st.info("No signals computed yet.")

        if error_results:
            with st.expander(f"⚠️ {len(error_results)} instrument(s) skipped"):
                for r in error_results:
                    st.write(f"- **{r.asset_name}**: {r.error}")

    # ---------------------------------------------------------------- TAB 2: CHART
    with tab_chart:
        if valid_results:
            focus_name = st.selectbox(
                "Select instrument", [r.asset_name for r in valid_results], key="chart_focus"
            )
            focus = next(r for r in valid_results if r.asset_name == focus_name)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Signal", focus.signal)
            c2.metric("Confidence", f"{focus.confidence:.1f}%")
            c3.metric("Z-Score", f"{focus.z_score:.2f}")
            c4.metric("Trend Strength (ADX-based)", f"{focus.trend_strength:.1f}")

            st.plotly_chart(
                ChartBuilder.price_chart(focus.df, focus, focus.asset_name),
                use_container_width=True,
            )
            st.plotly_chart(
                ChartBuilder.equity_curve_proxy(focus.df, focus.asset_name),
                use_container_width=True,
            )
        else:
            st.info("No chart data available.")

    # ---------------------------------------------------------------- TAB 3: DECAY
    with tab_decay:
        if valid_results:
            focus_name = st.selectbox(
                "Select instrument", [r.asset_name for r in valid_results], key="decay_focus"
            )
            focus = next(r for r in valid_results if r.asset_name == focus_name)

            st.plotly_chart(
                ChartBuilder.decay_curve(focus.acf_values, focus.alpha_half_life, focus.asset_name),
                use_container_width=True,
            )

            st.markdown(
                f"""
**Interpretation:** The Alpha Half-Life for **{focus.asset_name}** on the
**{cfg['timeframe']}** timeframe is approximately **{focus.alpha_half_life:.1f} bars**.
This is the holding period at which the predictive power of the current signal
is expected to have decayed to half its initial strength. Trade management
(the Take-Profit distance and position size) is scaled against this figure —
shorter half-lives compress the reward target and reduce size to reflect the
faster-fading edge; longer half-lives allow more room to run.
                """
            )

            st.subheader("Historical Holding-Period Efficiency")
            efficiency_df = pd.DataFrame({
                "Lag (bars)": list(range(1, len(focus.acf_values) + 1)),
                "Autocorrelation": [round(v, 4) for v in focus.acf_values],
                "|Autocorrelation|": [round(abs(v), 4) for v in focus.acf_values],
            })
            st.dataframe(efficiency_df, use_container_width=True)
        else:
            st.info("No decay data available.")

    # ---------------------------------------------------------------- TAB 4: RISK
    with tab_risk:
        st.subheader("Position Sizing Calculator")

        if valid_results:
            focus_name = st.selectbox(
                "Select instrument", [r.asset_name for r in valid_results], key="risk_focus"
            )
            focus = next(r for r in valid_results if r.asset_name == focus_name)

            if focus.signal == "NEUTRAL" or focus.entry is None:
                st.warning(
                    f"{focus.asset_name} is currently **NEUTRAL / NO TRADE** "
                    f"(confidence {focus.confidence:.1f}% < {MIN_CONFIDENCE_THRESHOLD:.0f}%). "
                    "No position sizing is generated for non-actionable signals."
                )
            else:
                sizing = RiskManager.position_size(
                    account_balance=cfg["account_balance"],
                    risk_pct=cfg["risk_pct"],
                    entry=focus.entry,
                    stop_loss=focus.stop_loss,
                    half_life=focus.alpha_half_life,
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Dollar Risk", f"${sizing['risk_amount']:,.2f}")
                col2.metric("Decay Weight", f"{sizing['decay_weight']*100:.0f}%")
                col3.metric("Position Size (units)", f"{sizing['units']:,.4f}")

                st.markdown(
                    f"""
**Basis:** Risking **{cfg['risk_pct']}%** of a **${cfg['account_balance']:,.0f}**
account = **${sizing['risk_amount']:,.2f}** at stake. Stop distance is
**{abs(focus.entry - focus.stop_loss):.5f}** price units, scaled down by a
**{sizing['decay_weight']*100:.0f}%** decay weight (Alpha Half-Life of
**{focus.alpha_half_life:.1f}** bars) since faster-decaying edges warrant
smaller size for the same dollar risk.
                    """
                )
        else:
            st.info("No instruments available for sizing.")

    st.caption(
        f"Data refreshed via yfinance · Cache TTL {CACHE_TTL_SECONDS}s · "
        f"Last run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · "
        "Educational / research tool — not financial advice."
    )


if __name__ == "__main__":
    main()
