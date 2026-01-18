import contextlib
import warnings
from llvmlite import ir
import numpy as np
import operator
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core import types, cgutils
from numba.core.errors import TypingError, NumbaTypeError, \
from .arrayobj import make_array, _empty_nd_impl, array_copy
from numba.np import numpy_support as np_support
def dot_2_vm(context, builder, sig, args):
    """
    np.dot(vector, matrix)
    """

    def dot_impl(a, b):
        m, = a.shape
        _m, n = b.shape
        if m == 0:
            return np.zeros((n,), a.dtype)
        out = np.empty((n,), a.dtype)
        return np.dot(a, b, out)
    res = context.compile_internal(builder, dot_impl, sig, args)
    return impl_ret_new_ref(context, builder, sig.return_type, res)