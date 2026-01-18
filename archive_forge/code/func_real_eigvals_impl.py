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
def real_eigvals_impl(a):
    """
        eigvals() implementation for real arrays.
        """
    n = a.shape[-1]
    if a.shape[-2] != n:
        msg = 'Last 2 dimensions of the array must be square.'
        raise np.linalg.LinAlgError(msg)
    _check_finite_matrix(a)
    acpy = _copy_to_fortran_order(a)
    ldvl = 1
    ldvr = 1
    wr = np.empty(n, dtype=a.dtype)
    if n == 0:
        return wr
    wi = np.empty(n, dtype=a.dtype)
    vl = np.empty(1, dtype=a.dtype)
    vr = np.empty(1, dtype=a.dtype)
    r = numba_ez_rgeev(kind, JOBVL, JOBVR, n, acpy.ctypes, n, wr.ctypes, wi.ctypes, vl.ctypes, ldvl, vr.ctypes, ldvr)
    _handle_err_maybe_convergence_problem(r)
    if np.any(wi):
        raise ValueError('eigvals() argument must not cause a domain change.')
    _dummy_liveness_func([acpy.size, vl.size, vr.size, wr.size, wi.size])
    return wr