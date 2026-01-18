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
def fisher_exact(table, alternative='two-sided'):
    """Perform a Fisher exact test on a 2x2 contingency table.

    The null hypothesis is that the true odds ratio of the populations
    underlying the observations is one, and the observations were sampled
    from these populations under a condition: the marginals of the
    resulting table must equal those of the observed table. The statistic
    returned is the unconditional maximum likelihood estimate of the odds
    ratio, and the p-value is the probability under the null hypothesis of
    obtaining a table at least as extreme as the one that was actually
    observed. There are other possible choices of statistic and two-sided
    p-value definition associated with Fisher's exact test; please see the
    Notes for more information.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table.  Elements must be non-negative integers.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis.
        The following options are available (default is 'two-sided'):

        * 'two-sided': the odds ratio of the underlying population is not one
        * 'less': the odds ratio of the underlying population is less than one
        * 'greater': the odds ratio of the underlying population is greater
          than one

        See the Notes for more details.

    Returns
    -------
    res : SignificanceResult
        An object containing attributes:

        statistic : float
            This is the prior odds ratio, not a posterior estimate.
        pvalue : float
            The probability under the null hypothesis of obtaining a
            table at least as extreme as the one that was actually observed.

    See Also
    --------
    chi2_contingency : Chi-square test of independence of variables in a
        contingency table.  This can be used as an alternative to
        `fisher_exact` when the numbers in the table are large.
    contingency.odds_ratio : Compute the odds ratio (sample or conditional
        MLE) for a 2x2 contingency table.
    barnard_exact : Barnard's exact test, which is a more powerful alternative
        than Fisher's exact test for 2x2 contingency tables.
    boschloo_exact : Boschloo's exact test, which is a more powerful
        alternative than Fisher's exact test for 2x2 contingency tables.

    Notes
    -----
    *Null hypothesis and p-values*

    The null hypothesis is that the true odds ratio of the populations
    underlying the observations is one, and the observations were sampled at
    random from these populations under a condition: the marginals of the
    resulting table must equal those of the observed table. Equivalently,
    the null hypothesis is that the input table is from the hypergeometric
    distribution with parameters (as used in `hypergeom`)
    ``M = a + b + c + d``, ``n = a + b`` and ``N = a + c``, where the
    input table is ``[[a, b], [c, d]]``.  This distribution has support
    ``max(0, N + n - M) <= x <= min(N, n)``, or, in terms of the values
    in the input table, ``min(0, a - d) <= x <= a + min(b, c)``.  ``x``
    can be interpreted as the upper-left element of a 2x2 table, so the
    tables in the distribution have form::

        [  x           n - x     ]
        [N - x    M - (n + N) + x]

    For example, if::

        table = [6  2]
                [1  4]

    then the support is ``2 <= x <= 7``, and the tables in the distribution
    are::

        [2 6]   [3 5]   [4 4]   [5 3]   [6 2]  [7 1]
        [5 0]   [4 1]   [3 2]   [2 3]   [1 4]  [0 5]

    The probability of each table is given by the hypergeometric distribution
    ``hypergeom.pmf(x, M, n, N)``.  For this example, these are (rounded to
    three significant digits)::

        x       2      3      4      5       6        7
        p  0.0163  0.163  0.408  0.326  0.0816  0.00466

    These can be computed with::

        >>> import numpy as np
        >>> from scipy.stats import hypergeom
        >>> table = np.array([[6, 2], [1, 4]])
        >>> M = table.sum()
        >>> n = table[0].sum()
        >>> N = table[:, 0].sum()
        >>> start, end = hypergeom.support(M, n, N)
        >>> hypergeom.pmf(np.arange(start, end+1), M, n, N)
        array([0.01631702, 0.16317016, 0.40792541, 0.32634033, 0.08158508,
               0.004662  ])

    The two-sided p-value is the probability that, under the null hypothesis,
    a random table would have a probability equal to or less than the
    probability of the input table.  For our example, the probability of
    the input table (where ``x = 6``) is 0.0816.  The x values where the
    probability does not exceed this are 2, 6 and 7, so the two-sided p-value
    is ``0.0163 + 0.0816 + 0.00466 ~= 0.10256``::

        >>> from scipy.stats import fisher_exact
        >>> res = fisher_exact(table, alternative='two-sided')
        >>> res.pvalue
        0.10256410256410257

    The one-sided p-value for ``alternative='greater'`` is the probability
    that a random table has ``x >= a``, which in our example is ``x >= 6``,
    or ``0.0816 + 0.00466 ~= 0.08626``::

        >>> res = fisher_exact(table, alternative='greater')
        >>> res.pvalue
        0.08624708624708627

    This is equivalent to computing the survival function of the
    distribution at ``x = 5`` (one less than ``x`` from the input table,
    because we want to include the probability of ``x = 6`` in the sum)::

        >>> hypergeom.sf(5, M, n, N)
        0.08624708624708627

    For ``alternative='less'``, the one-sided p-value is the probability
    that a random table has ``x <= a``, (i.e. ``x <= 6`` in our example),
    or ``0.0163 + 0.163 + 0.408 + 0.326 + 0.0816 ~= 0.9949``::

        >>> res = fisher_exact(table, alternative='less')
        >>> res.pvalue
        0.9953379953379957

    This is equivalent to computing the cumulative distribution function
    of the distribution at ``x = 6``:

        >>> hypergeom.cdf(6, M, n, N)
        0.9953379953379957

    *Odds ratio*

    The calculated odds ratio is different from the value computed by the
    R function ``fisher.test``.  This implementation returns the "sample"
    or "unconditional" maximum likelihood estimate, while ``fisher.test``
    in R uses the conditional maximum likelihood estimate.  To compute the
    conditional maximum likelihood estimate of the odds ratio, use
    `scipy.stats.contingency.odds_ratio`.

    References
    ----------
    .. [1] Fisher, Sir Ronald A, "The Design of Experiments:
           Mathematics of a Lady Tasting Tea." ISBN 978-0-486-41151-4, 1935.
    .. [2] "Fisher's exact test",
           https://en.wikipedia.org/wiki/Fisher's_exact_test
    .. [3] Emma V. Low et al. "Identifying the lowest effective dose of
           acetazolamide for the prophylaxis of acute mountain sickness:
           systematic review and meta-analysis."
           BMJ, 345, :doi:`10.1136/bmj.e6779`, 2012.

    Examples
    --------
    In [3]_, the effective dose of acetazolamide for the prophylaxis of acute
    mountain sickness was investigated. The study notably concluded:

        Acetazolamide 250 mg, 500 mg, and 750 mg daily were all efficacious for
        preventing acute mountain sickness. Acetazolamide 250 mg was the lowest
        effective dose with available evidence for this indication.

    The following table summarizes the results of the experiment in which
    some participants took a daily dose of acetazolamide 250 mg while others
    took a placebo.
    Cases of acute mountain sickness were recorded::

                                    Acetazolamide   Control/Placebo
        Acute mountain sickness            7           17
        No                                15            5


    Is there evidence that the acetazolamide 250 mg reduces the risk of
    acute mountain sickness?
    We begin by formulating a null hypothesis :math:`H_0`:

        The odds of experiencing acute mountain sickness are the same with
        the acetazolamide treatment as they are with placebo.

    Let's assess the plausibility of this hypothesis with
    Fisher's test.

    >>> from scipy.stats import fisher_exact
    >>> res = fisher_exact([[7, 17], [15, 5]], alternative='less')
    >>> res.statistic
    0.13725490196078433
    >>> res.pvalue
    0.0028841933752349743

    Using a significance level of 5%, we would reject the null hypothesis in
    favor of the alternative hypothesis: "The odds of experiencing acute
    mountain sickness with acetazolamide treatment are less than the odds of
    experiencing acute mountain sickness with placebo."

    .. note::

        Because the null distribution of Fisher's exact test is formed under
        the assumption that both row and column sums are fixed, the result of
        the test are conservative when applied to an experiment in which the
        row sums are not fixed.

        In this case, the column sums are fixed; there are 22 subjects in each
        group. But the number of cases of acute mountain sickness is not
        (and cannot be) fixed before conducting the experiment. It is a
        consequence.

        Boschloo's test does not depend on the assumption that the row sums
        are fixed, and consequently, it provides a more powerful test in this
        situation.

        >>> from scipy.stats import boschloo_exact
        >>> res = boschloo_exact([[7, 17], [15, 5]], alternative='less')
        >>> res.statistic
        0.0028841933752349743
        >>> res.pvalue
        0.0015141406667567101

        We verify that the p-value is less than with `fisher_exact`.

    """
    hypergeom = distributions.hypergeom
    c = np.asarray(table, dtype=np.int64)
    if not c.shape == (2, 2):
        raise ValueError('The input `table` must be of shape (2, 2).')
    if np.any(c < 0):
        raise ValueError('All values in `table` must be nonnegative.')
    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        return SignificanceResult(np.nan, 1.0)
    if c[1, 0] > 0 and c[0, 1] > 0:
        oddsratio = c[0, 0] * c[1, 1] / (c[1, 0] * c[0, 1])
    else:
        oddsratio = np.inf
    n1 = c[0, 0] + c[0, 1]
    n2 = c[1, 0] + c[1, 1]
    n = c[0, 0] + c[1, 0]

    def pmf(x):
        return hypergeom.pmf(x, n1 + n2, n1, n)
    if alternative == 'less':
        pvalue = hypergeom.cdf(c[0, 0], n1 + n2, n1, n)
    elif alternative == 'greater':
        pvalue = hypergeom.cdf(c[0, 1], n1 + n2, n1, c[0, 1] + c[1, 1])
    elif alternative == 'two-sided':
        mode = int((n + 1) * (n1 + 1) / (n1 + n2 + 2))
        pexact = hypergeom.pmf(c[0, 0], n1 + n2, n1, n)
        pmode = hypergeom.pmf(mode, n1 + n2, n1, n)
        epsilon = 1e-14
        gamma = 1 + epsilon
        if np.abs(pexact - pmode) / np.maximum(pexact, pmode) <= epsilon:
            return SignificanceResult(oddsratio, 1.0)
        elif c[0, 0] < mode:
            plower = hypergeom.cdf(c[0, 0], n1 + n2, n1, n)
            if hypergeom.pmf(n, n1 + n2, n1, n) > pexact * gamma:
                return SignificanceResult(oddsratio, plower)
            guess = _binary_search(lambda x: -pmf(x), -pexact * gamma, mode, n)
            pvalue = plower + hypergeom.sf(guess, n1 + n2, n1, n)
        else:
            pupper = hypergeom.sf(c[0, 0] - 1, n1 + n2, n1, n)
            if hypergeom.pmf(0, n1 + n2, n1, n) > pexact * gamma:
                return SignificanceResult(oddsratio, pupper)
            guess = _binary_search(pmf, pexact * gamma, 0, mode)
            pvalue = pupper + hypergeom.cdf(guess, n1 + n2, n1, n)
    else:
        msg = "`alternative` should be one of {'two-sided', 'less', 'greater'}"
        raise ValueError(msg)
    pvalue = min(pvalue, 1.0)
    return SignificanceResult(oddsratio, pvalue)