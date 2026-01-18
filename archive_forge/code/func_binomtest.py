from math import sqrt
import numpy as np
from scipy._lib._util import _validate_int
from scipy.optimize import brentq
from scipy.special import ndtri
from ._discrete_distns import binom
from ._common import ConfidenceInterval
def binomtest(k, n, p=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.

    The binomial test [1]_ is a test of the null hypothesis that the
    probability of success in a Bernoulli experiment is `p`.

    Details of the test can be found in many texts on statistics, such
    as section 24.5 of [2]_.

    Parameters
    ----------
    k : int
        The number of successes.
    n : int
        The number of trials.
    p : float, optional
        The hypothesized probability of success, i.e. the expected
        proportion of successes.  The value must be in the interval
        ``0 <= p <= 1``. The default value is ``p = 0.5``.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Indicates the alternative hypothesis. The default value is
        'two-sided'.

    Returns
    -------
    result : `~scipy.stats._result_classes.BinomTestResult` instance
        The return value is an object with the following attributes:

        k : int
            The number of successes (copied from `binomtest` input).
        n : int
            The number of trials (copied from `binomtest` input).
        alternative : str
            Indicates the alternative hypothesis specified in the input
            to `binomtest`.  It will be one of ``'two-sided'``, ``'greater'``,
            or ``'less'``.
        statistic : float
            The estimate of the proportion of successes.
        pvalue : float
            The p-value of the hypothesis test.

        The object has the following methods:

        proportion_ci(confidence_level=0.95, method='exact') :
            Compute the confidence interval for ``statistic``.

    Notes
    -----
    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] Binomial test, https://en.wikipedia.org/wiki/Binomial_test
    .. [2] Jerrold H. Zar, Biostatistical Analysis (fifth edition),
           Prentice Hall, Upper Saddle River, New Jersey USA (2010)

    Examples
    --------
    >>> from scipy.stats import binomtest

    A car manufacturer claims that no more than 10% of their cars are unsafe.
    15 cars are inspected for safety, 3 were found to be unsafe. Test the
    manufacturer's claim:

    >>> result = binomtest(3, n=15, p=0.1, alternative='greater')
    >>> result.pvalue
    0.18406106910639114

    The null hypothesis cannot be rejected at the 5% level of significance
    because the returned p-value is greater than the critical value of 5%.

    The test statistic is equal to the estimated proportion, which is simply
    ``3/15``:

    >>> result.statistic
    0.2

    We can use the `proportion_ci()` method of the result to compute the
    confidence interval of the estimate:

    >>> result.proportion_ci(confidence_level=0.95)
    ConfidenceInterval(low=0.05684686759024681, high=1.0)

    """
    k = _validate_int(k, 'k', minimum=0)
    n = _validate_int(n, 'n', minimum=1)
    if k > n:
        raise ValueError(f'k ({k}) must not be greater than n ({n}).')
    if not 0 <= p <= 1:
        raise ValueError(f'p ({p}) must be in range [0,1]')
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError(f"alternative ('{alternative}') not recognized; \nmust be 'two-sided', 'less' or 'greater'")
    if alternative == 'less':
        pval = binom.cdf(k, n, p)
    elif alternative == 'greater':
        pval = binom.sf(k - 1, n, p)
    else:
        d = binom.pmf(k, n, p)
        rerr = 1 + 1e-07
        if k == p * n:
            pval = 1.0
        elif k < p * n:
            ix = _binary_search_for_binom_tst(lambda x1: -binom.pmf(x1, n, p), -d * rerr, np.ceil(p * n), n)
            y = n - ix + int(d * rerr == binom.pmf(ix, n, p))
            pval = binom.cdf(k, n, p) + binom.sf(n - y, n, p)
        else:
            ix = _binary_search_for_binom_tst(lambda x1: binom.pmf(x1, n, p), d * rerr, 0, np.floor(p * n))
            y = ix + 1
            pval = binom.cdf(y - 1, n, p) + binom.sf(k - 1, n, p)
        pval = min(1.0, pval)
    result = BinomTestResult(k=k, n=n, alternative=alternative, statistic=k / n, pvalue=pval)
    return result