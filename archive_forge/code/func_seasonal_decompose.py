import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tools.validation import PandasWrapper, array_like
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.stl.mstl import MSTL
from statsmodels.tsa.tsatools import freq_to_period
def seasonal_decompose(x, model='additive', filt=None, period=None, two_sided=True, extrapolate_trend=0):
    """
    Seasonal decomposition using moving averages.

    Parameters
    ----------
    x : array_like
        Time series. If 2d, individual series are in columns. x must contain 2
        complete cycles.
    model : {"additive", "multiplicative"}, optional
        Type of seasonal component. Abbreviations are accepted.
    filt : array_like, optional
        The filter coefficients for filtering out the seasonal component.
        The concrete moving average method used in filtering is determined by
        two_sided.
    period : int, optional
        Period of the series (e.g., 1 for annual, 4 for quarterly, etc). Must
        be used if x is not a pandas object or if the index of x does not have
        a frequency. Overrides default periodicity of x if x is a pandas
        object with a timeseries index.
    two_sided : bool, optional
        The moving average method used in filtering.
        If True (default), a centered moving average is computed using the
        filt. If False, the filter coefficients are for past values only.
    extrapolate_trend : int or 'freq', optional
        If set to > 0, the trend resulting from the convolution is
        linear least-squares extrapolated on both ends (or the single one
        if two_sided is False) considering this many (+1) closest points.
        If set to 'freq', use `freq` closest points. Setting this parameter
        results in no NaN values in trend or resid components.

    Returns
    -------
    DecomposeResult
        A object with seasonal, trend, and resid attributes.

    See Also
    --------
    statsmodels.tsa.filters.bk_filter.bkfilter
        Baxter-King filter.
    statsmodels.tsa.filters.cf_filter.cffilter
        Christiano-Fitzgerald asymmetric, random walk filter.
    statsmodels.tsa.filters.hp_filter.hpfilter
        Hodrick-Prescott filter.
    statsmodels.tsa.filters.convolution_filter
        Linear filtering via convolution.
    statsmodels.tsa.seasonal.STL
        Season-Trend decomposition using LOESS.

    Notes
    -----
    This is a naive decomposition. More sophisticated methods should
    be preferred.

    The additive model is Y[t] = T[t] + S[t] + e[t]

    The multiplicative model is Y[t] = T[t] * S[t] * e[t]

    The results are obtained by first estimating the trend by applying
    a convolution filter to the data. The trend is then removed from the
    series and the average of this de-trended series for each period is
    the returned seasonal component.
    """
    pfreq = period
    pw = PandasWrapper(x)
    if period is None:
        pfreq = getattr(getattr(x, 'index', None), 'inferred_freq', None)
    x = array_like(x, 'x', maxdim=2)
    nobs = len(x)
    if not np.all(np.isfinite(x)):
        raise ValueError('This function does not handle missing values')
    if model.startswith('m'):
        if np.any(x <= 0):
            raise ValueError('Multiplicative seasonality is not appropriate for zero and negative values')
    if period is None:
        if pfreq is not None:
            pfreq = freq_to_period(pfreq)
            period = pfreq
        else:
            raise ValueError('You must specify a period or x must be a pandas object with a PeriodIndex or a DatetimeIndex with a freq not set to None')
    if x.shape[0] < 2 * pfreq:
        raise ValueError(f'x must have 2 complete cycles requires {2 * pfreq} observations. x only has {x.shape[0]} observation(s)')
    if filt is None:
        if period % 2 == 0:
            filt = np.array([0.5] + [1] * (period - 1) + [0.5]) / period
        else:
            filt = np.repeat(1.0 / period, period)
    nsides = int(two_sided) + 1
    trend = convolution_filter(x, filt, nsides)
    if extrapolate_trend == 'freq':
        extrapolate_trend = period - 1
    if extrapolate_trend > 0:
        trend = _extrapolate_trend(trend, extrapolate_trend + 1)
    if model.startswith('m'):
        detrended = x / trend
    else:
        detrended = x - trend
    period_averages = seasonal_mean(detrended, period)
    if model.startswith('m'):
        period_averages /= np.mean(period_averages, axis=0)
    else:
        period_averages -= np.mean(period_averages, axis=0)
    seasonal = np.tile(period_averages.T, nobs // period + 1).T[:nobs]
    if model.startswith('m'):
        resid = x / seasonal / trend
    else:
        resid = detrended - seasonal
    results = []
    for s, name in zip((seasonal, trend, resid, x), ('seasonal', 'trend', 'resid', None)):
        results.append(pw.wrap(s.squeeze(), columns=name))
    return DecomposeResult(seasonal=results[0], trend=results[1], resid=results[2], observed=results[3])