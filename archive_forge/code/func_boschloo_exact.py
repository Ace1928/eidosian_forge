from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import chi2, norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py
def boschloo_exact(table, alternative='two-sided', n=32):
    """Perform Boschloo's exact test on a 2x2 contingency table.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements should be non-negative integers.

    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Please see explanations in the Notes section below.

    n : int, optional
        Number of sampling points used in the construction of the sampling
        method. Note that this argument will automatically be converted to
        the next higher power of 2 since `scipy.stats.qmc.Sobol` is used to
        select sample points. Default is 32. Must be positive. In most cases,
        32 points is enough to reach good precision. More points comes at
        performance cost.

    Returns
    -------
    ber : BoschlooExactResult
        A result object with the following attributes.

        statistic : float
            The statistic used in Boschloo's test; that is, the p-value
            from Fisher's exact test.

        pvalue : float
            P-value, the probability of obtaining a distribution at least as
            extreme as the one that was actually observed, assuming that the
            null hypothesis is true.

    See Also
    --------
    chi2_contingency : Chi-square test of independence of variables in a
        contingency table.
    fisher_exact : Fisher exact test on a 2x2 contingency table.
    barnard_exact : Barnard's exact test, which is a more powerful alternative
        than Fisher's exact test for 2x2 contingency tables.

    Notes
    -----
    Boschloo's test is an exact test used in the analysis of contingency
    tables. It examines the association of two categorical variables, and
    is a uniformly more powerful alternative to Fisher's exact test
    for 2x2 contingency tables.

    Boschloo's exact test uses the p-value of Fisher's exact test as a
    statistic, and Boschloo's p-value is the probability under the null
    hypothesis of observing such an extreme value of this statistic.

    Let's define :math:`X_0` a 2x2 matrix representing the observed sample,
    where each column stores the binomial experiment, as in the example
    below. Let's also define :math:`p_1, p_2` the theoretical binomial
    probabilities for  :math:`x_{11}` and :math:`x_{12}`. When using
    Boschloo exact test, we can assert three different alternative hypotheses:

    - :math:`H_0 : p_1=p_2` versus :math:`H_1 : p_1 < p_2`,
      with `alternative` = "less"

    - :math:`H_0 : p_1=p_2` versus :math:`H_1 : p_1 > p_2`,
      with `alternative` = "greater"

    - :math:`H_0 : p_1=p_2` versus :math:`H_1 : p_1 \\neq p_2`,
      with `alternative` = "two-sided" (default)

    There are multiple conventions for computing a two-sided p-value when the
    null distribution is asymmetric. Here, we apply the convention that the
    p-value of a two-sided test is twice the minimum of the p-values of the
    one-sided tests (clipped to 1.0). Note that `fisher_exact` follows a
    different convention, so for a given `table`, the statistic reported by
    `boschloo_exact` may differ from the p-value reported by `fisher_exact`
    when ``alternative='two-sided'``.

    .. versionadded:: 1.7.0

    References
    ----------
    .. [1] R.D. Boschloo. "Raised conditional level of significance for the
       2 x 2-table when testing the equality of two probabilities",
       Statistica Neerlandica, 24(1), 1970

    .. [2] "Boschloo's test", Wikipedia,
       https://en.wikipedia.org/wiki/Boschloo%27s_test

    .. [3] Lise M. Saari et al. "Employee attitudes and job satisfaction",
       Human Resource Management, 43(4), 395-407, 2004,
       :doi:`10.1002/hrm.20032`.

    Examples
    --------
    In the following example, we consider the article "Employee
    attitudes and job satisfaction" [3]_
    which reports the results of a survey from 63 scientists and 117 college
    professors. Of the 63 scientists, 31 said they were very satisfied with
    their jobs, whereas 74 of the college professors were very satisfied
    with their work. Is this significant evidence that college
    professors are happier with their work than scientists?
    The following table summarizes the data mentioned above::

                         college professors   scientists
        Very Satisfied   74                     31
        Dissatisfied     43                     32

    When working with statistical hypothesis testing, we usually use a
    threshold probability or significance level upon which we decide
    to reject the null hypothesis :math:`H_0`. Suppose we choose the common
    significance level of 5%.

    Our alternative hypothesis is that college professors are truly more
    satisfied with their work than scientists. Therefore, we expect
    :math:`p_1` the proportion of very satisfied college professors to be
    greater than :math:`p_2`, the proportion of very satisfied scientists.
    We thus call `boschloo_exact` with the ``alternative="greater"`` option:

    >>> import scipy.stats as stats
    >>> res = stats.boschloo_exact([[74, 31], [43, 32]], alternative="greater")
    >>> res.statistic
    0.0483...
    >>> res.pvalue
    0.0355...

    Under the null hypothesis that scientists are happier in their work than
    college professors, the probability of obtaining test
    results at least as extreme as the observed data is approximately 3.55%.
    Since this p-value is less than our chosen significance level, we have
    evidence to reject :math:`H_0` in favor of the alternative hypothesis.

    """
    hypergeom = distributions.hypergeom
    if n <= 0:
        raise ValueError(f'Number of points `n` must be strictly positive, found {n!r}')
    table = np.asarray(table, dtype=np.int64)
    if not table.shape == (2, 2):
        raise ValueError('The input `table` must be of shape (2, 2).')
    if np.any(table < 0):
        raise ValueError('All values in `table` must be nonnegative.')
    if 0 in table.sum(axis=0):
        return BoschlooExactResult(np.nan, np.nan)
    total_col_1, total_col_2 = table.sum(axis=0)
    total = total_col_1 + total_col_2
    x1 = np.arange(total_col_1 + 1, dtype=np.int64).reshape(1, -1)
    x2 = np.arange(total_col_2 + 1, dtype=np.int64).reshape(-1, 1)
    x1_sum_x2 = x1 + x2
    if alternative == 'less':
        pvalues = hypergeom.cdf(x1, total, x1_sum_x2, total_col_1).T
    elif alternative == 'greater':
        pvalues = hypergeom.cdf(x2, total, x1_sum_x2, total_col_2).T
    elif alternative == 'two-sided':
        boschloo_less = boschloo_exact(table, alternative='less', n=n)
        boschloo_greater = boschloo_exact(table, alternative='greater', n=n)
        res = boschloo_less if boschloo_less.pvalue < boschloo_greater.pvalue else boschloo_greater
        pvalue = np.clip(2 * res.pvalue, a_min=0, a_max=1)
        return BoschlooExactResult(res.statistic, pvalue)
    else:
        msg = f'`alternative` should be one of {('two-sided', 'less', 'greater')}, found {alternative!r}'
        raise ValueError(msg)
    fisher_stat = pvalues[table[0, 0], table[0, 1]]
    index_arr = pvalues <= fisher_stat * (1 + 1e-13)
    x1, x2, x1_sum_x2 = (x1.T, x2.T, x1_sum_x2.T)
    x1_log_comb = _compute_log_combinations(total_col_1)
    x2_log_comb = _compute_log_combinations(total_col_2)
    x1_sum_x2_log_comb = x1_log_comb[x1] + x2_log_comb[x2]
    result = shgo(_get_binomial_log_p_value_with_nuisance_param, args=(x1_sum_x2, x1_sum_x2_log_comb, index_arr), bounds=((0, 1),), n=n, sampling_method='sobol')
    p_value = np.clip(np.exp(-result.fun), a_min=0, a_max=1)
    return BoschlooExactResult(fisher_stat, p_value)