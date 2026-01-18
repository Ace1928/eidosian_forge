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
def quantile_test_iv(x, q, p, alternative):
    x = np.atleast_1d(x)
    message = '`x` must be a one-dimensional array of numbers.'
    if x.ndim != 1 or not np.issubdtype(x.dtype, np.number):
        raise ValueError(message)
    q = np.array(q)[()]
    message = '`q` must be a scalar.'
    if q.ndim != 0 or not np.issubdtype(q.dtype, np.number):
        raise ValueError(message)
    p = np.array(p)[()]
    message = '`p` must be a float strictly between 0 and 1.'
    if p.ndim != 0 or p >= 1 or p <= 0:
        raise ValueError(message)
    alternatives = {'two-sided', 'less', 'greater'}
    message = f'`alternative` must be one of {alternatives}'
    if alternative not in alternatives:
        raise ValueError(message)
    return (x, q, p, alternative)