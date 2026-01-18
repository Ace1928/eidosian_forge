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