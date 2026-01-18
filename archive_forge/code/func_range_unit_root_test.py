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
def range_unit_root_test(x, store=False):
    """
    Range unit-root test for stationarity.

    Computes the Range Unit-Root (RUR) test for the null
    hypothesis that x is stationary.

    Parameters
    ----------
    x : array_like, 1d
        The data series to test.
    store : bool
        If True, then a result instance is returned additionally to
        the RUR statistic (default is False).

    Returns
    -------
    rur_stat : float
        The RUR test statistic.
    p_value : float
        The p-value of the test. The p-value is interpolated from
        Table 1 in Aparicio et al. (2006), and a boundary point
        is returned if the test statistic is outside the table of
        critical values, that is, if the p-value is outside the
        interval (0.01, 0.1).
    crit : dict
        The critical values at 10%, 5%, 2.5% and 1%. Based on
        Aparicio et al. (2006).
    resstore : (optional) instance of ResultStore
        An instance of a dummy class with results attached as attributes.

    Notes
    -----
    The p-values are interpolated from
    Table 1 of Aparicio et al. (2006). If the computed statistic is
    outside the table of critical values, then a warning message is
    generated.

    Missing values are not handled.

    References
    ----------
    .. [1] Aparicio, F., Escribano A., Sipols, A.E. (2006). Range Unit-Root (RUR)
        tests: robust against nonlinearities, error distributions, structural breaks
        and outliers. Journal of Time Series Analysis, 27 (4): 545-576.
    """
    x = array_like(x, 'x')
    store = bool_like(store, 'store')
    nobs = x.shape[0]
    if nobs != x.size:
        raise ValueError(f'x of shape {x.shape} not understood')
    pvals = [0.01, 0.025, 0.05, 0.1, 0.9, 0.95]
    n = np.array([25, 50, 100, 150, 200, 250, 500, 1000, 2000, 3000, 4000, 5000])
    crit = np.array([[0.6626, 0.8126, 0.9192, 1.0712, 2.4863, 2.7312], [0.7977, 0.9274, 1.0478, 1.1964, 2.6821, 2.9613], [0.907, 1.0243, 1.1412, 1.2888, 2.8317, 3.1393], [0.9543, 1.0768, 1.1869, 1.3294, 2.8915, 3.2049], [0.9833, 1.0984, 1.2101, 1.3494, 2.9308, 3.2482], [0.9982, 1.1137, 1.2242, 1.3632, 2.9571, 3.2842], [1.0494, 1.1643, 1.2712, 1.4076, 3.0207, 3.3584], [1.0846, 1.1959, 1.2988, 1.4344, 3.0653, 3.4073], [1.1121, 1.22, 1.323, 1.4556, 3.0948, 3.4439], [1.1204, 1.2295, 1.3303, 1.4656, 3.1054, 3.4632], [1.1309, 1.2347, 1.3378, 1.4693, 3.1165, 3.4717], [1.1377, 1.2402, 1.3408, 1.4729, 3.1252, 3.4807]])
    inter_crit = np.zeros((1, crit.shape[1]))
    for i in range(crit.shape[1]):
        f = interp1d(n, crit[:, i])
        inter_crit[0, i] = f(nobs)
    xs = pd.Series(x)
    exp_max = xs.expanding(1).max().shift(1)
    exp_min = xs.expanding(1).min().shift(1)
    count = (xs > exp_max).sum() + (xs < exp_min).sum()
    rur_stat = count / np.sqrt(len(x))
    k = len(pvals) - 1
    for i in range(len(pvals) - 1, -1, -1):
        if rur_stat < inter_crit[0, i]:
            k = i
        else:
            break
    p_value = pvals[k]
    warn_msg = 'The test statistic is outside of the range of p-values available in the\nlook-up table. The actual p-value is {direction} than the p-value returned.\n'
    direction = ''
    if p_value == pvals[-1]:
        direction = 'smaller'
    elif p_value == pvals[0]:
        direction = 'larger'
    if direction:
        warnings.warn(warn_msg.format(direction=direction), InterpolationWarning, stacklevel=2)
    crit_dict = {'10%': inter_crit[0, 3], '5%': inter_crit[0, 2], '2.5%': inter_crit[0, 1], '1%': inter_crit[0, 0]}
    if store:
        from statsmodels.stats.diagnostic import ResultsStore
        rstore = ResultsStore()
        rstore.nobs = nobs
        rstore.H0 = 'The series is not stationary'
        rstore.HA = 'The series is stationary'
        return (rur_stat, p_value, crit_dict, rstore)
    else:
        return (rur_stat, p_value, crit_dict)