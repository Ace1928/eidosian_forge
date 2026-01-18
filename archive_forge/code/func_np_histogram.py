import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@overload(np.histogram)
def np_histogram(a, bins=10, range=None):
    if isinstance(bins, (int, types.Integer)):
        if range in (None, types.none):
            inf = float('inf')

            def histogram_impl(a, bins=10, range=None):
                bin_min = inf
                bin_max = -inf
                for view in np.nditer(a):
                    v = view.item()
                    if bin_min > v:
                        bin_min = v
                    if bin_max < v:
                        bin_max = v
                return np.histogram(a, bins, (bin_min, bin_max))
        else:

            def histogram_impl(a, bins=10, range=None):
                if bins <= 0:
                    raise ValueError('histogram(): `bins` should be a positive integer')
                bin_min, bin_max = range
                if not bin_min <= bin_max:
                    raise ValueError('histogram(): max must be larger than min in range parameter')
                hist = np.zeros(bins, np.intp)
                if bin_max > bin_min:
                    bin_ratio = bins / (bin_max - bin_min)
                    for view in np.nditer(a):
                        v = view.item()
                        b = math.floor((v - bin_min) * bin_ratio)
                        if 0 <= b < bins:
                            hist[int(b)] += 1
                        elif v == bin_max:
                            hist[bins - 1] += 1
                bins_array = np.linspace(bin_min, bin_max, bins + 1)
                return (hist, bins_array)
    else:

        def histogram_impl(a, bins=10, range=None):
            nbins = len(bins) - 1
            for i in _range(nbins):
                if not bins[i] <= bins[i + 1]:
                    raise ValueError('histogram(): bins must increase monotonically')
            bin_min = bins[0]
            bin_max = bins[nbins]
            hist = np.zeros(nbins, np.intp)
            if nbins > 0:
                for view in np.nditer(a):
                    v = view.item()
                    if not bin_min <= v <= bin_max:
                        continue
                    lo = 0
                    hi = nbins - 1
                    while lo < hi:
                        mid = lo + hi + 1 >> 1
                        if v < bins[mid]:
                            hi = mid - 1
                        else:
                            lo = mid
                    hist[lo] += 1
            return (hist, bins)
    return histogram_impl