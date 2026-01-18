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
def adfuller(x, maxlag: int | None=None, regression='c', autolag='AIC', store=False, regresults=False):
    """
    Augmented Dickey-Fuller unit root test.

    The Augmented Dickey-Fuller test can be used to test for a unit root in a
    univariate process in the presence of serial correlation.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    maxlag : {None, int}
        Maximum lag which is included in test, default value of
        12*(nobs/100)^{1/4} is used when ``None``.
    regression : {"c","ct","ctt","n"}
        Constant and trend order to include in regression.

        * "c" : constant only (default).
        * "ct" : constant and trend.
        * "ctt" : constant, and linear and quadratic trend.
        * "n" : no constant, no trend.

    autolag : {"AIC", "BIC", "t-stat", None}
        Method to use when automatically determining the lag length among the
        values 0, 1, ..., maxlag.

        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
        * If None, then the number of included lags is set to maxlag.
    store : bool
        If True, then a result instance is returned additionally to
        the adf statistic. Default is False.
    regresults : bool, optional
        If True, the full regression results are returned. Default is False.

    Returns
    -------
    adf : float
        The test statistic.
    pvalue : float
        MacKinnon's approximate p-value based on MacKinnon (1994, 2010).
    usedlag : int
        The number of lags used.
    nobs : int
        The number of observations used for the ADF regression and calculation
        of the critical values.
    critical values : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels. Based on MacKinnon (2010).
    icbest : float
        The maximized information criterion if autolag is not None.
    resstore : ResultStore, optional
        A dummy class with results attached as attributes.

    Notes
    -----
    The null hypothesis of the Augmented Dickey-Fuller is that there is a unit
    root, with the alternative that there is no unit root. If the pvalue is
    above a critical size, then we cannot reject that there is a unit root.

    The p-values are obtained through regression surface approximation from
    MacKinnon 1994, but using the updated 2010 tables. If the p-value is close
    to significant, then the critical values should be used to judge whether
    to reject the null.

    The autolag option and maxlag for it are described in Greene.

    See the notebook `Stationarity and detrending (ADF/KPSS)
    <../examples/notebooks/generated/stationarity_detrending_adf_kpss.html>`__
    for an overview.

    References
    ----------
    .. [1] W. Green.  "Econometric Analysis," 5th ed., Pearson, 2003.

    .. [2] Hamilton, J.D.  "Time Series Analysis".  Princeton, 1994.

    .. [3] MacKinnon, J.G. 1994.  "Approximate asymptotic distribution functions for
        unit-root and cointegration tests.  `Journal of Business and Economic
        Statistics` 12, 167-76.

    .. [4] MacKinnon, J.G. 2010. "Critical Values for Cointegration Tests."  Queen"s
        University, Dept of Economics, Working Papers.  Available at
        http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    x = array_like(x, 'x')
    maxlag = int_like(maxlag, 'maxlag', optional=True)
    regression = string_like(regression, 'regression', options=('c', 'ct', 'ctt', 'n'))
    autolag = string_like(autolag, 'autolag', optional=True, options=('aic', 'bic', 't-stat'))
    store = bool_like(store, 'store')
    regresults = bool_like(regresults, 'regresults')
    if x.max() == x.min():
        raise ValueError('Invalid input, x is constant')
    if regresults:
        store = True
    trenddict = {None: 'n', 0: 'c', 1: 'ct', 2: 'ctt'}
    if regression is None or isinstance(regression, int):
        regression = trenddict[regression]
    regression = regression.lower()
    nobs = x.shape[0]
    ntrend = len(regression) if regression != 'n' else 0
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * np.power(nobs / 100.0, 1 / 4.0)))
        maxlag = min(nobs // 2 - ntrend - 1, maxlag)
        if maxlag < 0:
            raise ValueError('sample size is too short to use selected regression component')
    elif maxlag > nobs // 2 - ntrend - 1:
        raise ValueError('maxlag must be less than (nobs/2 - 1 - ntrend) where n trend is the number of included deterministic regressors')
    xdiff = np.diff(x)
    xdall = lagmat(xdiff[:, None], maxlag, trim='both', original='in')
    nobs = xdall.shape[0]
    xdall[:, 0] = x[-nobs - 1:-1]
    xdshort = xdiff[-nobs:]
    if store:
        from statsmodels.stats.diagnostic import ResultsStore
        resstore = ResultsStore()
    if autolag:
        if regression != 'n':
            fullRHS = add_trend(xdall, regression, prepend=True)
        else:
            fullRHS = xdall
        startlag = fullRHS.shape[1] - xdall.shape[1] + 1
        if not regresults:
            icbest, bestlag = _autolag(OLS, xdshort, fullRHS, startlag, maxlag, autolag)
        else:
            icbest, bestlag, alres = _autolag(OLS, xdshort, fullRHS, startlag, maxlag, autolag, regresults=regresults)
            resstore.autolag_results = alres
        bestlag -= startlag
        xdall = lagmat(xdiff[:, None], bestlag, trim='both', original='in')
        nobs = xdall.shape[0]
        xdall[:, 0] = x[-nobs - 1:-1]
        xdshort = xdiff[-nobs:]
        usedlag = bestlag
    else:
        usedlag = maxlag
        icbest = None
    if regression != 'n':
        resols = OLS(xdshort, add_trend(xdall[:, :usedlag + 1], regression)).fit()
    else:
        resols = OLS(xdshort, xdall[:, :usedlag + 1]).fit()
    adfstat = resols.tvalues[0]
    pvalue = mackinnonp(adfstat, regression=regression, N=1)
    critvalues = mackinnoncrit(N=1, regression=regression, nobs=nobs)
    critvalues = {'1%': critvalues[0], '5%': critvalues[1], '10%': critvalues[2]}
    if store:
        resstore.resols = resols
        resstore.maxlag = maxlag
        resstore.usedlag = usedlag
        resstore.adfstat = adfstat
        resstore.critvalues = critvalues
        resstore.nobs = nobs
        resstore.H0 = 'The coefficient on the lagged level equals 1 - unit root'
        resstore.HA = 'The coefficient on the lagged level < 1 - stationary'
        resstore.icbest = icbest
        resstore._str = 'Augmented Dickey-Fuller Test Results'
        return (adfstat, pvalue, critvalues, resstore)
    elif not autolag:
        return (adfstat, pvalue, usedlag, nobs, critvalues)
    else:
        return (adfstat, pvalue, usedlag, nobs, critvalues, icbest)