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
def kpss(x, regression: Literal['c', 'ct']='c', nlags: Literal['auto', 'legacy'] | int='auto', store: bool=False) -> tuple[float, float, int, dict[str, float]]:
    """
    Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

    Computes the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for the null
    hypothesis that x is level or trend stationary.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    regression : str{"c", "ct"}
        The null hypothesis for the KPSS test.

        * "c" : The data is stationary around a constant (default).
        * "ct" : The data is stationary around a trend.
    nlags : {str, int}, optional
        Indicates the number of lags to be used. If "auto" (default), lags
        is calculated using the data-dependent method of Hobijn et al. (1998).
        See also Andrews (1991), Newey & West (1994), and Schwert (1989). If
        set to "legacy",  uses int(12 * (n / 100)**(1 / 4)) , as outlined in
        Schwert (1989).
    store : bool
        If True, then a result instance is returned additionally to
        the KPSS statistic (default is False).

    Returns
    -------
    kpss_stat : float
        The KPSS test statistic.
    p_value : float
        The p-value of the test. The p-value is interpolated from
        Table 1 in Kwiatkowski et al. (1992), and a boundary point
        is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the
        interval (0.01, 0.1).
    lags : int
        The truncation lag parameter.
    crit : dict
        The critical values at 10%, 5%, 2.5% and 1%. Based on
        Kwiatkowski et al. (1992).
    resstore : (optional) instance of ResultStore
        An instance of a dummy class with results attached as attributes.

    Notes
    -----
    To estimate sigma^2 the Newey-West estimator is used. If lags is "legacy",
    the truncation lag parameter is set to int(12 * (n / 100) ** (1 / 4)),
    as outlined in Schwert (1989). The p-values are interpolated from
    Table 1 of Kwiatkowski et al. (1992). If the computed statistic is
    outside the table of critical values, then a warning message is
    generated.

    Missing values are not handled.

    See the notebook `Stationarity and detrending (ADF/KPSS)
    <../examples/notebooks/generated/stationarity_detrending_adf_kpss.html>`__
    for an overview.

    References
    ----------
    .. [1] Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation
       consistent covariance matrix estimation. Econometrica, 59: 817-858.

    .. [2] Hobijn, B., Frances, B.H., & Ooms, M. (2004). Generalizations of the
       KPSS-test for stationarity. Statistica Neerlandica, 52: 483-502.

    .. [3] Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., & Shin, Y. (1992).
       Testing the null hypothesis of stationarity against the alternative of a
       unit root. Journal of Econometrics, 54: 159-178.

    .. [4] Newey, W.K., & West, K.D. (1994). Automatic lag selection in
       covariance matrix estimation. Review of Economic Studies, 61: 631-653.

    .. [5] Schwert, G. W. (1989). Tests for unit roots: A Monte Carlo
       investigation. Journal of Business and Economic Statistics, 7 (2):
       147-159.
    """
    x = array_like(x, 'x')
    regression = string_like(regression, 'regression', options=('c', 'ct'))
    store = bool_like(store, 'store')
    nobs = x.shape[0]
    hypo = regression
    if nobs != x.size:
        raise ValueError(f'x of shape {x.shape} not understood')
    if hypo == 'ct':
        resids = OLS(x, add_constant(np.arange(1, nobs + 1))).fit().resid
        crit = [0.119, 0.146, 0.176, 0.216]
    else:
        resids = x - x.mean()
        crit = [0.347, 0.463, 0.574, 0.739]
    if nlags == 'legacy':
        nlags = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        nlags = min(nlags, nobs - 1)
    elif nlags == 'auto' or nlags is None:
        if nlags is None:
            warnings.warn("None is not a valid value for nlags. It must be an integer, 'auto' or 'legacy'. None will raise starting in 0.14", FutureWarning, stacklevel=2)
        nlags = _kpss_autolag(resids, nobs)
        nlags = min(nlags, nobs - 1)
    elif isinstance(nlags, str):
        raise ValueError("nvals must be 'auto' or 'legacy' when not an int")
    else:
        nlags = int_like(nlags, 'nlags', optional=False)
        if nlags >= nobs:
            raise ValueError(f'lags ({nlags}) must be < number of observations ({nobs})')
    pvals = [0.1, 0.05, 0.025, 0.01]
    eta = np.sum(resids.cumsum() ** 2) / nobs ** 2
    s_hat = _sigma_est_kpss(resids, nobs, nlags)
    kpss_stat = eta / s_hat
    p_value = np.interp(kpss_stat, crit, pvals)
    warn_msg = 'The test statistic is outside of the range of p-values available in the\nlook-up table. The actual p-value is {direction} than the p-value returned.\n'
    if p_value == pvals[-1]:
        warnings.warn(warn_msg.format(direction='smaller'), InterpolationWarning, stacklevel=2)
    elif p_value == pvals[0]:
        warnings.warn(warn_msg.format(direction='greater'), InterpolationWarning, stacklevel=2)
    crit_dict = {'10%': crit[0], '5%': crit[1], '2.5%': crit[2], '1%': crit[3]}
    if store:
        from statsmodels.stats.diagnostic import ResultsStore
        rstore = ResultsStore()
        rstore.lags = nlags
        rstore.nobs = nobs
        stationary_type = 'level' if hypo == 'c' else 'trend'
        rstore.H0 = f'The series is {stationary_type} stationary'
        rstore.HA = f'The series is not {stationary_type} stationary'
        return (kpss_stat, p_value, crit_dict, rstore)
    else:
        return (kpss_stat, p_value, nlags, crit_dict)