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
def svd_impl(a, full_matrices=1):
    n = a.shape[-1]
    m = a.shape[-2]
    if n == 0 or m == 0:
        raise np.linalg.LinAlgError('Arrays cannot be empty')
    _check_finite_matrix(a)
    acpy = _copy_to_fortran_order(a)
    ldu = m
    minmn = min(m, n)
    if full_matrices:
        JOBZ = JOBZ_A
        ucol = m
        ldvt = n
    else:
        JOBZ = JOBZ_S
        ucol = minmn
        ldvt = minmn
    u = np.empty((ucol, ldu), dtype=a.dtype)
    s = np.empty(minmn, dtype=s_dtype)
    vt = np.empty((n, ldvt), dtype=a.dtype)
    r = numba_ez_gesdd(kind, JOBZ, m, n, acpy.ctypes, m, s.ctypes, u.ctypes, ldu, vt.ctypes, ldvt)
    _handle_err_maybe_convergence_problem(r)
    _dummy_liveness_func([acpy.size, vt.size, u.size, s.size])
    return (u.T, s, vt.T)