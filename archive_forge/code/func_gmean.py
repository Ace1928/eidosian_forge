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
@_axis_nan_policy_factory(lambda x: x, n_samples=1, n_outputs=1, too_small=0, paired=True, result_to_tuple=lambda x: (x,), kwd_samples=['weights'])
def gmean(a, axis=0, dtype=None, weights=None):
    """Compute the weighted geometric mean along the specified axis.

    The weighted geometric mean of the array :math:`a_i` associated to weights
    :math:`w_i` is:

    .. math::

        \\exp \\left( \\frac{ \\sum_{i=1}^n w_i \\ln a_i }{ \\sum_{i=1}^n w_i }
                   \\right) \\, ,

    and, with equal weights, it gives:

    .. math::

        \\sqrt[n]{ \\prod_{i=1}^n a_i } \\, .

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int or None, optional
        Axis along which the geometric mean is computed. Default is 0.
        If None, compute over the whole array `a`.
    dtype : dtype, optional
        Type to which the input arrays are cast before the calculation is
        performed.
    weights : array_like, optional
        The `weights` array must be broadcastable to the same shape as `a`.
        Default is None, which gives each value a weight of 1.0.

    Returns
    -------
    gmean : ndarray
        See `dtype` parameter above.

    See Also
    --------
    numpy.mean : Arithmetic average
    numpy.average : Weighted average
    hmean : Harmonic mean

    References
    ----------
    .. [1] "Weighted Geometric Mean", *Wikipedia*,
           https://en.wikipedia.org/wiki/Weighted_geometric_mean.
    .. [2] Grossman, J., Grossman, M., Katz, R., "Averages: A New Approach",
           Archimedes Foundation, 1983

    Examples
    --------
    >>> from scipy.stats import gmean
    >>> gmean([1, 4])
    2.0
    >>> gmean([1, 2, 3, 4, 5, 6, 7])
    3.3800151591412964
    >>> gmean([1, 4, 7], weights=[3, 1, 3])
    2.80668351922014

    """
    a = np.asarray(a, dtype=dtype)
    if weights is not None:
        weights = np.asarray(weights, dtype=dtype)
    with np.errstate(divide='ignore'):
        log_a = np.log(a)
    return np.exp(np.average(log_a, axis=axis, weights=weights))