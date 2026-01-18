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
def dot_3_mv_check_args(a, b, out):
    m, _n = a.shape
    n, = b.shape
    if n != _n:
        raise ValueError('incompatible array sizes for np.dot(a, b) (matrix * vector)')
    if out.shape != (m,):
        raise ValueError('incompatible output array size for np.dot(a, b, out) (matrix * vector)')