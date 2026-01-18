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
def lstsq_impl(a, b, rcond=-1.0):
    n = a.shape[-1]
    m = a.shape[-2]
    nrhs = _system_compute_nrhs(b)
    _check_finite_matrix(a)
    _check_finite_matrix(b)
    _system_check_non_empty(a, b)
    _system_check_dimensionally_valid(a, b)
    minmn = min(m, n)
    maxmn = max(m, n)
    acpy = _copy_to_fortran_order(a)
    bcpy = np.empty((nrhs, maxmn), dtype=np_dt).T
    _system_copy_in_b(bcpy, b, nrhs)
    s = np.empty(minmn, dtype=real_dtype)
    rank_ptr = np.empty(1, dtype=np.int32)
    r = numba_ez_gelsd(kind, m, n, nrhs, acpy.ctypes, m, bcpy.ctypes, maxmn, s.ctypes, rcond, rank_ptr.ctypes)
    _handle_err_maybe_convergence_problem(r)
    rank = rank_ptr[0]
    if rank < n or m <= n:
        res = np.empty(0, dtype=real_dtype)
    else:
        res = _lstsq_residual(bcpy, n, nrhs)
    x = _lstsq_solution(b, bcpy, n)
    _dummy_liveness_func([acpy.size, bcpy.size, s.size, rank_ptr.size])
    return (x, res, rank, s[:minmn])