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
@overload(np.digitize)
def np_digitize(x, bins, right=False):
    if isinstance(x, types.Array) and x.dtype in types.complex_domain:
        raise TypingError('x may not be complex')

    @register_jitable
    def _monotonicity(bins):
        if len(bins) == 0:
            return 1
        last_value = bins[0]
        i = 1
        while i < len(bins) and bins[i] == last_value:
            i += 1
        if i == len(bins):
            return 1
        next_value = bins[i]
        if last_value < next_value:
            for i in range(i + 1, len(bins)):
                last_value = next_value
                next_value = bins[i]
                if last_value > next_value:
                    return 0
            return 1
        else:
            for i in range(i + 1, len(bins)):
                last_value = next_value
                next_value = bins[i]
                if last_value < next_value:
                    return 0
            return -1

    def digitize_impl(x, bins, right=False):
        mono = _monotonicity(bins)
        if mono == 0:
            raise ValueError('bins must be monotonically increasing or decreasing')
        if right:
            if mono == -1:
                return len(bins) - np.searchsorted(bins[::-1], x, side='left')
            else:
                return np.searchsorted(bins, x, side='left')
        elif mono == -1:
            return len(bins) - np.searchsorted(bins[::-1], x, side='right')
        else:
            return np.searchsorted(bins, x, side='right')
    return digitize_impl