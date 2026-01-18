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
@_axis_nan_policy_factory(RanksumsResult, n_samples=2)
def ranksums(x, y, alternative='two-sided'):
    """Compute the Wilcoxon rank-sum statistic for two samples.

    The Wilcoxon rank-sum test tests the null hypothesis that two sets
    of measurements are drawn from the same distribution.  The alternative
    hypothesis is that values in one sample are more likely to be
    larger than the values in the other sample.

    This test should be used to compare two samples from continuous
    distributions.  It does not handle ties between measurements
    in x and y.  For tie-handling and an optional continuity correction
    see `scipy.stats.mannwhitneyu`.

    Parameters
    ----------
    x,y : array_like
        The data from the two samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        The following options are available:

        * 'two-sided': one of the distributions (underlying `x` or `y`) is
          stochastically greater than the other.
        * 'less': the distribution underlying `x` is stochastically less
          than the distribution underlying `y`.
        * 'greater': the distribution underlying `x` is stochastically greater
          than the distribution underlying `y`.

        .. versionadded:: 1.7.0

    Returns
    -------
    statistic : float
        The test statistic under the large-sample approximation that the
        rank sum statistic is normally distributed.
    pvalue : float
        The p-value of the test.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test

    Examples
    --------
    We can test the hypothesis that two independent unequal-sized samples are
    drawn from the same distribution with computing the Wilcoxon rank-sum
    statistic.

    >>> import numpy as np
    >>> from scipy.stats import ranksums
    >>> rng = np.random.default_rng()
    >>> sample1 = rng.uniform(-1, 1, 200)
    >>> sample2 = rng.uniform(-0.5, 1.5, 300) # a shifted distribution
    >>> ranksums(sample1, sample2)
    RanksumsResult(statistic=-7.887059,
                   pvalue=3.09390448e-15) # may vary
    >>> ranksums(sample1, sample2, alternative='less')
    RanksumsResult(statistic=-7.750585297581713,
                   pvalue=4.573497606342543e-15) # may vary
    >>> ranksums(sample1, sample2, alternative='greater')
    RanksumsResult(statistic=-7.750585297581713,
                   pvalue=0.9999999999999954) # may vary

    The p-value of less than ``0.05`` indicates that this test rejects the
    hypothesis at the 5% significance level.

    """
    x, y = map(np.asarray, (x, y))
    n1 = len(x)
    n2 = len(y)
    alldata = np.concatenate((x, y))
    ranked = rankdata(alldata)
    x = ranked[:n1]
    s = np.sum(x, axis=0)
    expected = n1 * (n1 + n2 + 1) / 2.0
    z = (s - expected) / np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    z, prob = _normtest_finish(z, alternative)
    return RanksumsResult(z, prob)