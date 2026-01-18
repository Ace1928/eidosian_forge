from functools import partial
import numpy as np
from scipy import stats
from statsmodels.tools.validation import string_like
from ._lilliefors_critical_values import (critical_values,
from .tabledist import TableDist
def kstest_fit(x, dist='norm', pvalmethod='table'):
    """
    Test assumed normal or exponential distribution using Lilliefors' test.

    Lilliefors' test is a Kolmogorov-Smirnov test with estimated parameters.

    Parameters
    ----------
    x : array_like, 1d
        Data to test.
    dist : {'norm', 'exp'}, optional
        The assumed distribution.
    pvalmethod : {'approx', 'table'}, optional
        The method used to compute the p-value of the test statistic. In
        general, 'table' is preferred and makes use of a very large simulation.
        'approx' is only valid for normality. if `dist = 'exp'` `table` is
        always used. 'approx' uses the approximation formula of Dalal and
        Wilkinson, valid for pvalues < 0.1. If the pvalue is larger than 0.1,
        then the result of `table` is returned.

    Returns
    -------
    ksstat : float
        Kolmogorov-Smirnov test statistic with estimated mean and variance.
    pvalue : float
        If the pvalue is lower than some threshold, e.g. 0.05, then we can
        reject the Null hypothesis that the sample comes from a normal
        distribution.

    Notes
    -----
    'table' uses an improved table based on 10,000,000 simulations. The
    critical values are approximated using
    log(cv_alpha) = b_alpha + c[0] log(n) + c[1] log(n)**2
    where cv_alpha is the critical value for a test with size alpha,
    b_alpha is an alpha-specific intercept term and c[1] and c[2] are
    coefficients that are shared all alphas.
    Values in the table are linearly interpolated. Values outside the
    range are be returned as bounds, 0.990 for large and 0.001 for small
    pvalues.

    For implementation details, see  lilliefors_critical_value_simulation.py in
    the test directory.
    """
    pvalmethod = string_like(pvalmethod, 'pvalmethod', options=('approx', 'table'))
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    elif x.ndim != 1:
        raise ValueError('Invalid parameter `x`: must be a one-dimensional array-like or a single-column DataFrame')
    nobs = len(x)
    if dist == 'norm':
        z = (x - x.mean()) / x.std(ddof=1)
        test_d = stats.norm.cdf
        lilliefors_table = lilliefors_table_norm
    elif dist == 'exp':
        z = x / x.mean()
        test_d = stats.expon.cdf
        lilliefors_table = lilliefors_table_expon
        pvalmethod = 'table'
    else:
        raise ValueError("Invalid dist parameter, must be 'norm' or 'exp'")
    min_nobs = 4 if dist == 'norm' else 3
    if nobs < min_nobs:
        raise ValueError('Test for distribution {} requires at least {} observations'.format(dist, min_nobs))
    d_ks = ksstat(z, test_d, alternative='two_sided')
    if pvalmethod == 'approx':
        pval = pval_lf(d_ks, nobs)
        if pval > 0.1:
            pval = lilliefors_table.prob(d_ks, nobs)
    else:
        pval = lilliefors_table.prob(d_ks, nobs)
    return (d_ks, pval)