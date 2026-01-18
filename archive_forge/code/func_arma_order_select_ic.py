from __future__ import annotations
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import lzip
from statsmodels.compat.scipy import _next_regular
from typing import Literal, Union
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
def arma_order_select_ic(y, max_ar=4, max_ma=2, ic='bic', trend='c', model_kw=None, fit_kw=None):
    """
    Compute information criteria for many ARMA models.

    Parameters
    ----------
    y : array_like
        Array of time-series data.
    max_ar : int
        Maximum number of AR lags to use. Default 4.
    max_ma : int
        Maximum number of MA lags to use. Default 2.
    ic : str, list
        Information criteria to report. Either a single string or a list
        of different criteria is possible.
    trend : str
        The trend to use when fitting the ARMA models.
    model_kw : dict
        Keyword arguments to be passed to the ``ARMA`` model.
    fit_kw : dict
        Keyword arguments to be passed to ``ARMA.fit``.

    Returns
    -------
    Bunch
        Dict-like object with attribute access. Each ic is an attribute with a
        DataFrame for the results. The AR order used is the row index. The ma
        order used is the column index. The minimum orders are available as
        ``ic_min_order``.

    Notes
    -----
    This method can be used to tentatively identify the order of an ARMA
    process, provided that the time series is stationary and invertible. This
    function computes the full exact MLE estimate of each model and can be,
    therefore a little slow. An implementation using approximate estimates
    will be provided in the future. In the meantime, consider passing
    {method : "css"} to fit_kw.

    Examples
    --------

    >>> from statsmodels.tsa.arima_process import arma_generate_sample
    >>> import statsmodels.api as sm
    >>> import numpy as np

    >>> arparams = np.array([.75, -.25])
    >>> maparams = np.array([.65, .35])
    >>> arparams = np.r_[1, -arparams]
    >>> maparam = np.r_[1, maparams]
    >>> nobs = 250
    >>> np.random.seed(2014)
    >>> y = arma_generate_sample(arparams, maparams, nobs)
    >>> res = sm.tsa.arma_order_select_ic(y, ic=["aic", "bic"], trend="n")
    >>> res.aic_min_order
    >>> res.bic_min_order
    """
    max_ar = int_like(max_ar, 'max_ar')
    max_ma = int_like(max_ma, 'max_ma')
    trend = string_like(trend, 'trend', options=('n', 'c'))
    model_kw = dict_like(model_kw, 'model_kw', optional=True)
    fit_kw = dict_like(fit_kw, 'fit_kw', optional=True)
    ar_range = [i for i in range(max_ar + 1)]
    ma_range = [i for i in range(max_ma + 1)]
    if isinstance(ic, str):
        ic = [ic]
    elif not isinstance(ic, (list, tuple)):
        raise ValueError('Need a list or a tuple for ic if not a string.')
    results = np.zeros((len(ic), max_ar + 1, max_ma + 1))
    model_kw = {} if model_kw is None else model_kw
    fit_kw = {} if fit_kw is None else fit_kw
    y_arr = array_like(y, 'y', contiguous=True)
    for ar in ar_range:
        for ma in ma_range:
            mod = _safe_arma_fit(y_arr, (ar, 0, ma), model_kw, trend, fit_kw)
            if mod is None:
                results[:, ar, ma] = np.nan
                continue
            for i, criteria in enumerate(ic):
                results[i, ar, ma] = getattr(mod, criteria)
    dfs = [pd.DataFrame(res, columns=ma_range, index=ar_range) for res in results]
    res = dict(zip(ic, dfs))
    min_res = {}
    for i, result in res.items():
        delta = np.ascontiguousarray(np.abs(result.min().min() - result))
        ncols = delta.shape[1]
        loc = np.argmin(delta)
        min_res.update({i + '_min_order': (loc // ncols, loc % ncols)})
    res.update(min_res)
    return Bunch(**res)