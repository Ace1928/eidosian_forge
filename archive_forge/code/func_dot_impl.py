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
def dot_impl(a, b):
    m, n = a.shape
    _n, = b.shape
    if n == 0:
        return np.zeros((m,), a.dtype)
    out = np.empty((m,), a.dtype)
    return np.dot(a, b, out)