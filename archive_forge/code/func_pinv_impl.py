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
def pinv_impl(a, rcond=1e-15):
    n = a.shape[-1]
    m = a.shape[-2]
    _check_finite_matrix(a)
    acpy = _copy_to_fortran_order(a)
    if m == 0 or n == 0:
        return acpy.T.ravel().reshape(a.shape).T
    minmn = min(m, n)
    u = np.empty((minmn, m), dtype=a.dtype)
    s = np.empty(minmn, dtype=s_dtype)
    vt = np.empty((n, minmn), dtype=a.dtype)
    r = numba_ez_gesdd(kind, JOB, m, n, acpy.ctypes, m, s.ctypes, u.ctypes, m, vt.ctypes, minmn)
    _handle_err_maybe_convergence_problem(r)
    cut_at = s[0] * rcond
    cut_idx = 0
    for k in range(minmn):
        if s[k] > cut_at:
            s[k] = 1.0 / s[k]
            cut_idx = k
    cut_idx += 1
    if m >= n:
        for i in range(n):
            for j in range(cut_idx):
                vt[i, j] = vt[i, j] * s[j]
    else:
        for i in range(cut_idx):
            s_local = s[i]
            for j in range(minmn):
                u[i, j] = u[i, j] * s_local
    r = numba_xxgemm(kind, TRANSA, TRANSB, n, m, cut_idx, one.ctypes, vt.ctypes, minmn, u.ctypes, m, zero.ctypes, acpy.ctypes, n)
    _dummy_liveness_func([acpy.size, vt.size, u.size, s.size, one.size, zero.size])
    return acpy.T.ravel().reshape(a.shape).T