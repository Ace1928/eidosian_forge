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
def percentileofscore(a, score, kind='rank', nan_policy='propagate'):
    """Compute the percentile rank of a score relative to a list of scores.

    A `percentileofscore` of, for example, 80% means that 80% of the
    scores in `a` are below the given score. In the case of gaps or
    ties, the exact definition depends on the optional keyword, `kind`.

    Parameters
    ----------
    a : array_like
        Array to which `score` is compared.
    score : array_like
        Scores to compute percentiles for.
    kind : {'rank', 'weak', 'strict', 'mean'}, optional
        Specifies the interpretation of the resulting score.
        The following options are available (default is 'rank'):

          * 'rank': Average percentage ranking of score.  In case of multiple
            matches, average the percentage rankings of all matching scores.
          * 'weak': This kind corresponds to the definition of a cumulative
            distribution function.  A percentileofscore of 80% means that 80%
            of values are less than or equal to the provided score.
          * 'strict': Similar to "weak", except that only values that are
            strictly less than the given score are counted.
          * 'mean': The average of the "weak" and "strict" scores, often used
            in testing.  See https://en.wikipedia.org/wiki/Percentile_rank
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Specifies how to treat `nan` values in `a`.
        The following options are available (default is 'propagate'):

          * 'propagate': returns nan (for each value in `score`).
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values

    Returns
    -------
    pcos : float
        Percentile-position of score (0-100) relative to `a`.

    See Also
    --------
    numpy.percentile
    scipy.stats.scoreatpercentile, scipy.stats.rankdata

    Examples
    --------
    Three-quarters of the given values lie below a given score:

    >>> import numpy as np
    >>> from scipy import stats
    >>> stats.percentileofscore([1, 2, 3, 4], 3)
    75.0

    With multiple matches, note how the scores of the two matches, 0.6
    and 0.8 respectively, are averaged:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], 3)
    70.0

    Only 2/5 values are strictly less than 3:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], 3, kind='strict')
    40.0

    But 4/5 values are less than or equal to 3:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], 3, kind='weak')
    80.0

    The average between the weak and the strict scores is:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], 3, kind='mean')
    60.0

    Score arrays (of any dimensionality) are supported:

    >>> stats.percentileofscore([1, 2, 3, 3, 4], [2, 3])
    array([40., 70.])

    The inputs can be infinite:

    >>> stats.percentileofscore([-np.inf, 0, 1, np.inf], [1, 2, np.inf])
    array([75., 75., 100.])

    If `a` is empty, then the resulting percentiles are all `nan`:

    >>> stats.percentileofscore([], [1, 2])
    array([nan, nan])
    """
    a = np.asarray(a)
    n = len(a)
    score = np.asarray(score)
    cna, npa = _contains_nan(a, nan_policy, use_summation=False)
    cns, nps = _contains_nan(score, nan_policy, use_summation=False)
    if (cna or cns) and nan_policy == 'raise':
        raise ValueError('The input contains nan values')
    if cns:
        score = ma.masked_where(np.isnan(score), score)
    if cna:
        if nan_policy == 'omit':
            a = ma.masked_where(np.isnan(a), a)
            n = a.count()
        if nan_policy == 'propagate':
            n = 0
    if n == 0:
        perct = np.full_like(score, np.nan, dtype=np.float64)
    else:
        score = score[..., None]

        def count(x):
            return np.count_nonzero(x, -1)
        if kind == 'rank':
            left = count(a < score)
            right = count(a <= score)
            plus1 = left < right
            perct = (left + right + plus1) * (50.0 / n)
        elif kind == 'strict':
            perct = count(a < score) * (100.0 / n)
        elif kind == 'weak':
            perct = count(a <= score) * (100.0 / n)
        elif kind == 'mean':
            left = count(a < score)
            right = count(a <= score)
            perct = (left + right) * (50.0 / n)
        else:
            raise ValueError("kind can only be 'rank', 'strict', 'weak' or 'mean'")
    perct = ma.filled(perct, np.nan)
    if perct.ndim == 0:
        return perct[()]
    return perct