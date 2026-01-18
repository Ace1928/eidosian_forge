import warnings
import math
from math import gcd
from collections import namedtuple
import numpy as np
from numpy import array, asarray, ma
from scipy.spatial.distance import cdist
from scipy.ndimage import _measurements
from scipy._lib._util import (check_random_state, MapWrapper, _get_nan,
import scipy.special as special
from scipy import linalg
from . import distributions
from . import _mstats_basic as mstats_basic
from ._stats_mstats_common import (_find_repeats, linregress, theilslopes,
from ._stats import (_kendall_dis, _toint64, _weightedrankedtau,
from dataclasses import dataclass, field
from ._hypotests import _all_partitions
from ._stats_pythran import _compute_outer_prob_inside_method
from ._resampling import (MonteCarloMethod, PermutationMethod, BootstrapMethod,
from ._axis_nan_policy import (_axis_nan_policy_factory,
from ._binomtest import _binary_search_for_binom_tst as _binary_search
from scipy._lib._bunch import _make_tuple_bunch
from scipy import stats
from scipy.optimize import root_scalar
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy._lib._util import normalize_axis_index
from scipy._lib._util import float_factorial  # noqa: F401
from scipy.stats._mstats_basic import (  # noqa: F401
def quantile_test(x, *, q=0, p=0.5, alternative='two-sided'):
    """
    Perform a quantile test and compute a confidence interval of the quantile.

    This function tests the null hypothesis that `q` is the value of the
    quantile associated with probability `p` of the population underlying
    sample `x`. For example, with default parameters, it tests that the
    median of the population underlying `x` is zero. The function returns an
    object including the test statistic, a p-value, and a method for computing
    the confidence interval around the quantile.

    Parameters
    ----------
    x : array_like
        A one-dimensional sample.
    q : float, default: 0
        The hypothesized value of the quantile.
    p : float, default: 0.5
        The probability associated with the quantile; i.e. the proportion of
        the population less than `q` is `p`. Must be strictly between 0 and
        1.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the quantile associated with the probability `p`
          is not `q`.
        * 'less': the quantile associated with the probability `p` is less
          than `q`.
        * 'greater': the quantile associated with the probability `p` is
          greater than `q`.

    Returns
    -------
    result : QuantileTestResult
        An object with the following attributes:

        statistic : float
            One of two test statistics that may be used in the quantile test.
            The first test statistic, ``T1``, is the proportion of samples in
            `x` that are less than or equal to the hypothesized quantile
            `q`. The second test statistic, ``T2``, is the proportion of
            samples in `x` that are strictly less than the hypothesized
            quantile `q`.

            When ``alternative = 'greater'``, ``T1`` is used to calculate the
            p-value and ``statistic`` is set to ``T1``.

            When ``alternative = 'less'``, ``T2`` is used to calculate the
            p-value and ``statistic`` is set to ``T2``.

            When ``alternative = 'two-sided'``, both ``T1`` and ``T2`` are
            considered, and the one that leads to the smallest p-value is used.

        statistic_type : int
            Either `1` or `2` depending on which of ``T1`` or ``T2`` was
            used to calculate the p-value.

        pvalue : float
            The p-value associated with the given alternative.

        The object also has the following method:

        confidence_interval(confidence_level=0.95)
            Computes a confidence interval around the the
            population quantile associated with the probability `p`. The
            confidence interval is returned in a ``namedtuple`` with
            fields `low` and `high`.  Values are `nan` when there are
            not enough observations to compute the confidence interval at
            the desired confidence.

    Notes
    -----
    This test and its method for computing confidence intervals are
    non-parametric. They are valid if and only if the observations are i.i.d.

    The implementation of the test follows Conover [1]_. Two test statistics
    are considered.

    ``T1``: The number of observations in `x` less than or equal to `q`.

        ``T1 = (x <= q).sum()``

    ``T2``: The number of observations in `x` strictly less than `q`.

        ``T2 = (x < q).sum()``

    The use of two test statistics is necessary to handle the possibility that
    `x` was generated from a discrete or mixed distribution.

    The null hypothesis for the test is:

        H0: The :math:`p^{\\mathrm{th}}` population quantile is `q`.

    and the null distribution for each test statistic is
    :math:`\\mathrm{binom}\\left(n, p\\right)`. When ``alternative='less'``,
    the alternative hypothesis is:

        H1: The :math:`p^{\\mathrm{th}}` population quantile is less than `q`.

    and the p-value is the probability that the binomial random variable

    .. math::
        Y \\sim \\mathrm{binom}\\left(n, p\\right)

    is greater than or equal to the observed value ``T2``.

    When ``alternative='greater'``, the alternative hypothesis is:

        H1: The :math:`p^{\\mathrm{th}}` population quantile is greater than `q`

    and the p-value is the probability that the binomial random variable Y
    is less than or equal to the observed value ``T1``.

    When ``alternative='two-sided'``, the alternative hypothesis is

        H1: `q` is not the :math:`p^{\\mathrm{th}}` population quantile.

    and the p-value is twice the smaller of the p-values for the ``'less'``
    and ``'greater'`` cases. Both of these p-values can exceed 0.5 for the same
    data, so the value is clipped into the interval :math:`[0, 1]`.

    The approach for confidence intervals is attributed to Thompson [2]_ and
    later proven to be applicable to any set of i.i.d. samples [3]_. The
    computation is based on the observation that the probability of a quantile
    :math:`q` to be larger than any observations :math:`x_m (1\\leq m \\leq N)`
    can be computed as

    .. math::

        \\mathbb{P}(x_m \\leq q) = 1 - \\sum_{k=0}^{m-1} \\binom{N}{k}
        q^k(1-q)^{N-k}

    By default, confidence intervals are computed for a 95% confidence level.
    A common interpretation of a 95% confidence intervals is that if i.i.d.
    samples are drawn repeatedly from the same population and confidence
    intervals are formed each time, the confidence interval will contain the
    true value of the specified quantile in approximately 95% of trials.

    A similar function is available in the QuantileNPCI R package [4]_. The
    foundation is the same, but it computes the confidence interval bounds by
    doing interpolations between the sample values, whereas this function uses
    only sample values as bounds. Thus, ``quantile_test.confidence_interval``
    returns more conservative intervals (i.e., larger).

    The same computation of confidence intervals for quantiles is included in
    the confintr package [5]_.

    Two-sided confidence intervals are not guaranteed to be optimal; i.e.,
    there may exist a tighter interval that may contain the quantile of
    interest with probability larger than the confidence level.
    Without further assumption on the samples (e.g., the nature of the
    underlying distribution), the one-sided intervals are optimally tight.

    References
    ----------
    .. [1] W. J. Conover. Practical Nonparametric Statistics, 3rd Ed. 1999.
    .. [2] W. R. Thompson, "On Confidence Ranges for the Median and Other
       Expectation Distributions for Populations of Unknown Distribution
       Form," The Annals of Mathematical Statistics, vol. 7, no. 3,
       pp. 122-128, 1936, Accessed: Sep. 18, 2019. [Online]. Available:
       https://www.jstor.org/stable/2957563.
    .. [3] H. A. David and H. N. Nagaraja, "Order Statistics in Nonparametric
       Inference" in Order Statistics, John Wiley & Sons, Ltd, 2005, pp.
       159-170. Available:
       https://onlinelibrary.wiley.com/doi/10.1002/0471722162.ch7.
    .. [4] N. Hutson, A. Hutson, L. Yan, "QuantileNPCI: Nonparametric
       Confidence Intervals for Quantiles," R package,
       https://cran.r-project.org/package=QuantileNPCI
    .. [5] M. Mayer, "confintr: Confidence Intervals," R package,
       https://cran.r-project.org/package=confintr


    Examples
    --------

    Suppose we wish to test the null hypothesis that the median of a population
    is equal to 0.5. We choose a confidence level of 99%; that is, we will
    reject the null hypothesis in favor of the alternative if the p-value is
    less than 0.01.

    When testing random variates from the standard uniform distribution, which
    has a median of 0.5, we expect the data to be consistent with the null
    hypothesis most of the time.

    >>> import numpy as np
    >>> from scipy import stats
    >>> rng = np.random.default_rng(6981396440634228121)
    >>> rvs = stats.uniform.rvs(size=100, random_state=rng)
    >>> stats.quantile_test(rvs, q=0.5, p=0.5)
    QuantileTestResult(statistic=45, statistic_type=1, pvalue=0.36820161732669576)

    As expected, the p-value is not below our threshold of 0.01, so
    we cannot reject the null hypothesis.

    When testing data from the standard *normal* distribution, which has a
    median of 0, we would expect the null hypothesis to be rejected.

    >>> rvs = stats.norm.rvs(size=100, random_state=rng)
    >>> stats.quantile_test(rvs, q=0.5, p=0.5)
    QuantileTestResult(statistic=67, statistic_type=2, pvalue=0.0008737198369123724)

    Indeed, the p-value is lower than our threshold of 0.01, so we reject the
    null hypothesis in favor of the default "two-sided" alternative: the median
    of the population is *not* equal to 0.5.

    However, suppose we were to test the null hypothesis against the
    one-sided alternative that the median of the population is *greater* than
    0.5. Since the median of the standard normal is less than 0.5, we would not
    expect the null hypothesis to be rejected.

    >>> stats.quantile_test(rvs, q=0.5, p=0.5, alternative='greater')
    QuantileTestResult(statistic=67, statistic_type=1, pvalue=0.9997956114162866)

    Unsurprisingly, with a p-value greater than our threshold, we would not
    reject the null hypothesis in favor of the chosen alternative.

    The quantile test can be used for any quantile, not only the median. For
    example, we can test whether the third quartile of the distribution
    underlying the sample is greater than 0.6.

    >>> rvs = stats.uniform.rvs(size=100, random_state=rng)
    >>> stats.quantile_test(rvs, q=0.6, p=0.75, alternative='greater')
    QuantileTestResult(statistic=64, statistic_type=1, pvalue=0.00940696592998271)

    The p-value is lower than the threshold. We reject the null hypothesis in
    favor of the alternative: the third quartile of the distribution underlying
    our sample is greater than 0.6.

    `quantile_test` can also compute confidence intervals for any quantile.

    >>> rvs = stats.norm.rvs(size=100, random_state=rng)
    >>> res = stats.quantile_test(rvs, q=0.6, p=0.75)
    >>> ci = res.confidence_interval(confidence_level=0.95)
    >>> ci
    ConfidenceInterval(low=0.284491604437432, high=0.8912531024914844)

    When testing a one-sided alternative, the confidence interval contains
    all observations such that if passed as `q`, the p-value of the
    test would be greater than 0.05, and therefore the null hypothesis
    would not be rejected. For example:

    >>> rvs.sort()
    >>> q, p, alpha = 0.6, 0.75, 0.95
    >>> res = stats.quantile_test(rvs, q=q, p=p, alternative='less')
    >>> ci = res.confidence_interval(confidence_level=alpha)
    >>> for x in rvs[rvs <= ci.high]:
    ...     res = stats.quantile_test(rvs, q=x, p=p, alternative='less')
    ...     assert res.pvalue > 1-alpha
    >>> for x in rvs[rvs > ci.high]:
    ...     res = stats.quantile_test(rvs, q=x, p=p, alternative='less')
    ...     assert res.pvalue < 1-alpha

    Also, if a 95% confidence interval is repeatedly generated for random
    samples, the confidence interval will contain the true quantile value in
    approximately 95% of replications.

    >>> dist = stats.rayleigh() # our "unknown" distribution
    >>> p = 0.2
    >>> true_stat = dist.ppf(p) # the true value of the statistic
    >>> n_trials = 1000
    >>> quantile_ci_contains_true_stat = 0
    >>> for i in range(n_trials):
    ...     data = dist.rvs(size=100, random_state=rng)
    ...     res = stats.quantile_test(data, p=p)
    ...     ci = res.confidence_interval(0.95)
    ...     if ci[0] < true_stat < ci[1]:
    ...         quantile_ci_contains_true_stat += 1
    >>> quantile_ci_contains_true_stat >= 950
    True

    This works with any distribution and any quantile, as long as the samples
    are i.i.d.
    """
    X, x_star, p_star, H1 = quantile_test_iv(x, q, p, alternative)
    T1 = (X <= x_star).sum()
    T2 = (X < x_star).sum()
    n = len(X)
    Y = stats.binom(n=n, p=p_star)
    if H1 == 'less':
        pvalue = Y.sf(T2 - 1)
        statistic = T2
        statistic_type = 2
    elif H1 == 'greater':
        pvalue = Y.cdf(T1)
        statistic = T1
        statistic_type = 1
    elif H1 == 'two-sided':
        pvalues = [Y.cdf(T1), Y.sf(T2 - 1)]
        sorted_idx = np.argsort(pvalues)
        pvalue = np.clip(2 * pvalues[sorted_idx[0]], 0, 1)
        if sorted_idx[0]:
            statistic, statistic_type = (T2, 2)
        else:
            statistic, statistic_type = (T1, 1)
    return QuantileTestResult(statistic=statistic, statistic_type=statistic_type, pvalue=pvalue, _alternative=H1, _x=X, _p=p_star)