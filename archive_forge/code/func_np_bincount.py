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
@overload(np.bincount)
def np_bincount(a, weights=None, minlength=0):
    validate_1d_array_like('bincount', a)
    if not isinstance(a.dtype, types.Integer):
        return
    check_is_integer(minlength, 'minlength')
    if weights not in (None, types.none):
        validate_1d_array_like('bincount', weights)
        out_dtype = np.float64

        @register_jitable
        def validate_inputs(a, weights, minlength):
            if len(a) != len(weights):
                raise ValueError("bincount(): weights and list don't have the same length")

        @register_jitable
        def count_item(out, idx, val, weights):
            out[val] += weights[idx]
    else:
        out_dtype = types.intp

        @register_jitable
        def validate_inputs(a, weights, minlength):
            pass

        @register_jitable
        def count_item(out, idx, val, weights):
            out[val] += 1

    def bincount_impl(a, weights=None, minlength=0):
        validate_inputs(a, weights, minlength)
        if minlength < 0:
            raise ValueError("'minlength' must not be negative")
        n = len(a)
        a_max = a[0] if n > 0 else -1
        for i in range(1, n):
            if a[i] < 0:
                raise ValueError('bincount(): first argument must be non-negative')
            a_max = max(a_max, a[i])
        out_length = max(a_max + 1, minlength)
        out = np.zeros(out_length, out_dtype)
        for i in range(n):
            count_item(out, i, a[i], weights)
        return out
    return bincount_impl