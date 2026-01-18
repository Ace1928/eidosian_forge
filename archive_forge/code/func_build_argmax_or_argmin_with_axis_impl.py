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
def build_argmax_or_argmin_with_axis_impl(a, axis, flatten_impl):
    """
    Given a function that implements the logic for handling a flattened
    array, return the implementation function.
    """
    check_is_integer(axis, 'axis')
    retty = types.intp
    tuple_buffer = tuple(range(a.ndim))

    def impl(a, axis=None):
        if axis < 0:
            axis = a.ndim + axis
        if axis < 0 or axis >= a.ndim:
            raise ValueError('axis is out of bounds')
        if a.ndim == 1:
            return flatten_impl(a)
        tmp = tuple_buffer
        for i in range(axis, a.ndim - 1):
            tmp = tuple_setitem(tmp, i, i + 1)
        transpose_index = tuple_setitem(tmp, a.ndim - 1, axis)
        transposed_arr = a.transpose(transpose_index)
        m = transposed_arr.shape[-1]
        raveled = transposed_arr.ravel()
        assert raveled.size == a.size
        assert transposed_arr.size % m == 0
        out = np.empty(transposed_arr.size // m, retty)
        for i in range(out.size):
            out[i] = flatten_impl(raveled[i * m:(i + 1) * m])
        return out.reshape(transposed_arr.shape[:-1])
    return impl