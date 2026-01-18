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
def breakvar_heteroskedasticity_test(resid, subset_length=1 / 3, alternative='two-sided', use_f=True):
    """
    Test for heteroskedasticity of residuals

    Tests whether the sum-of-squares in the first subset of the sample is
    significantly different than the sum-of-squares in the last subset
    of the sample. Analogous to a Goldfeld-Quandt test. The null hypothesis
    is of no heteroskedasticity.

    Parameters
    ----------
    resid : array_like
        Residuals of a time series model.
        The shape is 1d (nobs,) or 2d (nobs, nvars).
    subset_length : {int, float}
        Length of the subsets to test (h in Notes below).
        If a float in 0 < subset_length < 1, it is interpreted as fraction.
        Default is 1/3.
    alternative : str, 'increasing', 'decreasing' or 'two-sided'
        This specifies the alternative for the p-value calculation. Default
        is two-sided.
    use_f : bool, optional
        Whether or not to compare against the asymptotic distribution
        (chi-squared) or the approximate small-sample distribution (F).
        Default is True (i.e. default is to compare against an F
        distribution).

    Returns
    -------
    test_statistic : {float, ndarray}
        Test statistic(s) H(h).
    p_value : {float, ndarray}
        p-value(s) of test statistic(s).

    Notes
    -----
    The null hypothesis is of no heteroskedasticity. That means different
    things depending on which alternative is selected:

    - Increasing: Null hypothesis is that the variance is not increasing
        throughout the sample; that the sum-of-squares in the later
        subsample is *not* greater than the sum-of-squares in the earlier
        subsample.
    - Decreasing: Null hypothesis is that the variance is not decreasing
        throughout the sample; that the sum-of-squares in the earlier
        subsample is *not* greater than the sum-of-squares in the later
        subsample.
    - Two-sided: Null hypothesis is that the variance is not changing
        throughout the sample. Both that the sum-of-squares in the earlier
        subsample is not greater than the sum-of-squares in the later
        subsample *and* that the sum-of-squares in the later subsample is
        not greater than the sum-of-squares in the earlier subsample.

    For :math:`h = [T/3]`, the test statistic is:

    .. math::

        H(h) = \\sum_{t=T-h+1}^T  \\tilde v_t^2
        \\Bigg / \\sum_{t=1}^{h} \\tilde v_t^2

    This statistic can be tested against an :math:`F(h,h)` distribution.
    Alternatively, :math:`h H(h)` is asymptotically distributed according
    to :math:`\\chi_h^2`; this second test can be applied by passing
    `use_f=False` as an argument.

    See section 5.4 of [1]_ for the above formula and discussion, as well
    as additional details.

    References
    ----------
    .. [1] Harvey, Andrew C. 1990. *Forecasting, Structural Time Series*
            *Models and the Kalman Filter.* Cambridge University Press.
    """
    squared_resid = np.asarray(resid, dtype=float) ** 2
    if squared_resid.ndim == 1:
        squared_resid = squared_resid.reshape(-1, 1)
    nobs = len(resid)
    if 0 < subset_length < 1:
        h = int(np.round(nobs * subset_length))
    elif type(subset_length) is int and subset_length >= 1:
        h = subset_length
    numer_resid = squared_resid[-h:]
    numer_dof = (~np.isnan(numer_resid)).sum(axis=0)
    numer_squared_sum = np.nansum(numer_resid, axis=0)
    for i, dof in enumerate(numer_dof):
        if dof < 2:
            warnings.warn('Early subset of data for variable %d has too few non-missing observations to calculate test statistic.' % i, stacklevel=2)
            numer_squared_sum[i] = np.nan
    denom_resid = squared_resid[:h]
    denom_dof = (~np.isnan(denom_resid)).sum(axis=0)
    denom_squared_sum = np.nansum(denom_resid, axis=0)
    for i, dof in enumerate(denom_dof):
        if dof < 2:
            warnings.warn('Later subset of data for variable %d has too few non-missing observations to calculate test statistic.' % i, stacklevel=2)
            denom_squared_sum[i] = np.nan
    test_statistic = numer_squared_sum / denom_squared_sum
    if use_f:
        from scipy.stats import f
        pval_lower = lambda test_statistics: f.cdf(test_statistics, numer_dof, denom_dof)
        pval_upper = lambda test_statistics: f.sf(test_statistics, numer_dof, denom_dof)
    else:
        from scipy.stats import chi2
        pval_lower = lambda test_statistics: chi2.cdf(numer_dof * test_statistics, denom_dof)
        pval_upper = lambda test_statistics: chi2.sf(numer_dof * test_statistics, denom_dof)
    alternative = alternative.lower()
    if alternative in ['i', 'inc', 'increasing']:
        p_value = pval_upper(test_statistic)
    elif alternative in ['d', 'dec', 'decreasing']:
        test_statistic = 1.0 / test_statistic
        p_value = pval_upper(test_statistic)
    elif alternative in ['2', '2-sided', 'two-sided']:
        p_value = 2 * np.minimum(pval_lower(test_statistic), pval_upper(test_statistic))
    else:
        raise ValueError('Invalid alternative.')
    if len(test_statistic) == 1:
        return (test_statistic[0], p_value[0])
    return (test_statistic, p_value)