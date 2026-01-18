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
def inv_impl(a):
    n = a.shape[-1]
    if a.shape[-2] != n:
        msg = 'Last 2 dimensions of the array must be square.'
        raise np.linalg.LinAlgError(msg)
    _check_finite_matrix(a)
    acpy = _copy_to_fortran_order(a)
    if n == 0:
        return acpy
    ipiv = np.empty(n, dtype=F_INT_nptype)
    r = numba_xxgetrf(kind, n, n, acpy.ctypes, n, ipiv.ctypes)
    _inv_err_handler(r)
    r = numba_xxgetri(kind, n, acpy.ctypes, n, ipiv.ctypes)
    _inv_err_handler(r)
    _dummy_liveness_func([acpy.size, ipiv.size])
    return acpy