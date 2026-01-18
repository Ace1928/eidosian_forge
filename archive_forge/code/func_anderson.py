from __future__ import annotations
import math
import warnings
from collections import namedtuple
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan
from ._ansari_swilk_statistics import gscale, swilk
from . import _stats_py
from ._fit import FitResult
from ._stats_py import find_repeats, _normtest_finish, SignificanceResult
from .contingency import chi2_contingency
from . import distributions
from ._distn_infrastructure import rv_generic
from ._hypotests import _get_wilcoxon_distr
from ._axis_nan_policy import _axis_nan_policy_factory
def anderson(x, dist='norm'):
    """Anderson-Darling test for data coming from a particular distribution.

    The Anderson-Darling test tests the null hypothesis that a sample is
    drawn from a population that follows a particular distribution.
    For the Anderson-Darling test, the critical values depend on
    which distribution is being tested against.  This function works
    for normal, exponential, logistic, weibull_min, or Gumbel (Extreme Value
    Type I) distributions.

    Parameters
    ----------
    x : array_like
        Array of sample data.
    dist : {'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l', 'gumbel_r', 'extreme1', 'weibull_min'}, optional
        The type of distribution to test against.  The default is 'norm'.
        The names 'extreme1', 'gumbel_l' and 'gumbel' are synonyms for the
        same distribution.

    Returns
    -------
    result : AndersonResult
        An object with the following attributes:

        statistic : float
            The Anderson-Darling test statistic.
        critical_values : list
            The critical values for this distribution.
        significance_level : list
            The significance levels for the corresponding critical values
            in percents.  The function returns critical values for a
            differing set of significance levels depending on the
            distribution that is being tested against.
        fit_result : `~scipy.stats._result_classes.FitResult`
            An object containing the results of fitting the distribution to
            the data.

    See Also
    --------
    kstest : The Kolmogorov-Smirnov test for goodness-of-fit.

    Notes
    -----
    Critical values provided are for the following significance levels:

    normal/exponential
        15%, 10%, 5%, 2.5%, 1%
    logistic
        25%, 10%, 5%, 2.5%, 1%, 0.5%
    gumbel_l / gumbel_r
        25%, 10%, 5%, 2.5%, 1%
    weibull_min
        50%, 25%, 15%, 10%, 5%, 2.5%, 1%, 0.5%

    If the returned statistic is larger than these critical values then
    for the corresponding significance level, the null hypothesis that
    the data come from the chosen distribution can be rejected.
    The returned statistic is referred to as 'A2' in the references.

    For `weibull_min`, maximum likelihood estimation is known to be
    challenging. If the test returns successfully, then the first order
    conditions for a maximum likehood estimate have been verified and
    the critical values correspond relatively well to the significance levels,
    provided that the sample is sufficiently large (>10 observations [7]).
    However, for some data - especially data with no left tail - `anderson`
    is likely to result in an error message. In this case, consider
    performing a custom goodness of fit test using
    `scipy.stats.monte_carlo_test`.

    References
    ----------
    .. [1] https://www.itl.nist.gov/div898/handbook/prc/section2/prc213.htm
    .. [2] Stephens, M. A. (1974). EDF Statistics for Goodness of Fit and
           Some Comparisons, Journal of the American Statistical Association,
           Vol. 69, pp. 730-737.
    .. [3] Stephens, M. A. (1976). Asymptotic Results for Goodness-of-Fit
           Statistics with Unknown Parameters, Annals of Statistics, Vol. 4,
           pp. 357-369.
    .. [4] Stephens, M. A. (1977). Goodness of Fit for the Extreme Value
           Distribution, Biometrika, Vol. 64, pp. 583-588.
    .. [5] Stephens, M. A. (1977). Goodness of Fit with Special Reference
           to Tests for Exponentiality , Technical Report No. 262,
           Department of Statistics, Stanford University, Stanford, CA.
    .. [6] Stephens, M. A. (1979). Tests of Fit for the Logistic Distribution
           Based on the Empirical Distribution Function, Biometrika, Vol. 66,
           pp. 591-595.
    .. [7] Richard A. Lockhart and Michael A. Stephens "Estimation and Tests of
           Fit for the Three-Parameter Weibull Distribution"
           Journal of the Royal Statistical Society.Series B(Methodological)
           Vol. 56, No. 3 (1994), pp. 491-500, Table 0.

    Examples
    --------
    Test the null hypothesis that a random sample was drawn from a normal
    distribution (with unspecified mean and standard deviation).

    >>> import numpy as np
    >>> from scipy.stats import anderson
    >>> rng = np.random.default_rng()
    >>> data = rng.random(size=35)
    >>> res = anderson(data)
    >>> res.statistic
    0.8398018749744764
    >>> res.critical_values
    array([0.527, 0.6  , 0.719, 0.839, 0.998])
    >>> res.significance_level
    array([15. , 10. ,  5. ,  2.5,  1. ])

    The value of the statistic (barely) exceeds the critical value associated
    with a significance level of 2.5%, so the null hypothesis may be rejected
    at a significance level of 2.5%, but not at a significance level of 1%.

    """
    dist = dist.lower()
    if dist in {'extreme1', 'gumbel'}:
        dist = 'gumbel_l'
    dists = {'norm', 'expon', 'gumbel_l', 'gumbel_r', 'logistic', 'weibull_min'}
    if dist not in dists:
        raise ValueError(f'Invalid distribution; dist must be in {dists}.')
    y = sort(x)
    xbar = np.mean(x, axis=0)
    N = len(y)
    if dist == 'norm':
        s = np.std(x, ddof=1, axis=0)
        w = (y - xbar) / s
        fit_params = (xbar, s)
        logcdf = distributions.norm.logcdf(w)
        logsf = distributions.norm.logsf(w)
        sig = array([15, 10, 5, 2.5, 1])
        critical = around(_Avals_norm / (1.0 + 4.0 / N - 25.0 / N / N), 3)
    elif dist == 'expon':
        w = y / xbar
        fit_params = (0, xbar)
        logcdf = distributions.expon.logcdf(w)
        logsf = distributions.expon.logsf(w)
        sig = array([15, 10, 5, 2.5, 1])
        critical = around(_Avals_expon / (1.0 + 0.6 / N), 3)
    elif dist == 'logistic':

        def rootfunc(ab, xj, N):
            a, b = ab
            tmp = (xj - a) / b
            tmp2 = exp(tmp)
            val = [np.sum(1.0 / (1 + tmp2), axis=0) - 0.5 * N, np.sum(tmp * (1.0 - tmp2) / (1 + tmp2), axis=0) + N]
            return array(val)
        sol0 = array([xbar, np.std(x, ddof=1, axis=0)])
        sol = optimize.fsolve(rootfunc, sol0, args=(x, N), xtol=1e-05)
        w = (y - sol[0]) / sol[1]
        fit_params = sol
        logcdf = distributions.logistic.logcdf(w)
        logsf = distributions.logistic.logsf(w)
        sig = array([25, 10, 5, 2.5, 1, 0.5])
        critical = around(_Avals_logistic / (1.0 + 0.25 / N), 3)
    elif dist == 'gumbel_r':
        xbar, s = distributions.gumbel_r.fit(x)
        w = (y - xbar) / s
        fit_params = (xbar, s)
        logcdf = distributions.gumbel_r.logcdf(w)
        logsf = distributions.gumbel_r.logsf(w)
        sig = array([25, 10, 5, 2.5, 1])
        critical = around(_Avals_gumbel / (1.0 + 0.2 / sqrt(N)), 3)
    elif dist == 'gumbel_l':
        xbar, s = distributions.gumbel_l.fit(x)
        w = (y - xbar) / s
        fit_params = (xbar, s)
        logcdf = distributions.gumbel_l.logcdf(w)
        logsf = distributions.gumbel_l.logsf(w)
        sig = array([25, 10, 5, 2.5, 1])
        critical = around(_Avals_gumbel / (1.0 + 0.2 / sqrt(N)), 3)
    elif dist == 'weibull_min':
        message = 'Critical values of the test statistic are given for the asymptotic distribution. These may not be accurate for samples with fewer than 10 observations. Consider using `scipy.stats.monte_carlo_test`.'
        if N < 10:
            warnings.warn(message, stacklevel=2)
        m, loc, scale = distributions.weibull_min.fit(y)
        m, loc, scale = _weibull_fit_check((m, loc, scale), y)
        fit_params = (m, loc, scale)
        logcdf = stats.weibull_min(*fit_params).logcdf(y)
        logsf = stats.weibull_min(*fit_params).logsf(y)
        c = 1 / m
        sig = array([0.5, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995])
        critical = _get_As_weibull(c)
        critical = np.round(critical + 0.0005, decimals=3)
    i = arange(1, N + 1)
    A2 = -N - np.sum((2 * i - 1.0) / N * (logcdf + logsf[::-1]), axis=0)
    message = '`anderson` successfully fit the distribution to the data.'
    res = optimize.OptimizeResult(success=True, message=message)
    res.x = np.array(fit_params)
    fit_result = FitResult(getattr(distributions, dist), y, discrete=False, res=res)
    return AndersonResult(A2, critical, sig, fit_result=fit_result)