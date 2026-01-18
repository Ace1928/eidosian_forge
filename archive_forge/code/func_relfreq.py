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
def relfreq(a, numbins=10, defaultreallimits=None, weights=None):
    """Return a relative frequency histogram, using the histogram function.

    A relative frequency  histogram is a mapping of the number of
    observations in each of the bins relative to the total of observations.

    Parameters
    ----------
    a : array_like
        Input array.
    numbins : int, optional
        The number of bins to use for the histogram. Default is 10.
    defaultreallimits : tuple (lower, upper), optional
        The lower and upper values for the range of the histogram.
        If no value is given, a range slightly larger than the range of the
        values in a is used. Specifically ``(a.min() - s, a.max() + s)``,
        where ``s = (1/2)(a.max() - a.min()) / (numbins - 1)``.
    weights : array_like, optional
        The weights for each value in `a`. Default is None, which gives each
        value a weight of 1.0

    Returns
    -------
    frequency : ndarray
        Binned values of relative frequency.
    lowerlimit : float
        Lower real limit.
    binsize : float
        Width of each bin.
    extrapoints : int
        Extra points.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> a = np.array([2, 4, 1, 2, 3, 2])
    >>> res = stats.relfreq(a, numbins=4)
    >>> res.frequency
    array([ 0.16666667, 0.5       , 0.16666667,  0.16666667])
    >>> np.sum(res.frequency)  # relative frequencies should add up to 1
    1.0

    Create a normal distribution with 1000 random values

    >>> samples = stats.norm.rvs(size=1000, random_state=rng)

    Calculate relative frequencies

    >>> res = stats.relfreq(samples, numbins=25)

    Calculate space of values for x

    >>> x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,
    ...                                  res.frequency.size)

    Plot relative frequency histogram

    >>> fig = plt.figure(figsize=(5, 4))
    >>> ax = fig.add_subplot(1, 1, 1)
    >>> ax.bar(x, res.frequency, width=res.binsize)
    >>> ax.set_title('Relative frequency histogram')
    >>> ax.set_xlim([x.min(), x.max()])

    >>> plt.show()

    """
    a = np.asanyarray(a)
    h, l, b, e = _histogram(a, numbins, defaultreallimits, weights=weights)
    h = h / a.shape[0]
    return RelfreqResult(h, l, b, e)