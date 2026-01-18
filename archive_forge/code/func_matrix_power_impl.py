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
def matrix_power_impl(a, n):
    if n == 0:
        A = np.zeros(a.shape, dtype=np_dtype)
        for k in range(a.shape[0]):
            A[k, k] = 1.0
        return A
    am, an = (a.shape[-1], a.shape[-2])
    if am != an:
        raise ValueError('input must be a square array')
    if am == 0:
        return a.copy()
    if n < 0:
        A = np.linalg.inv(a).copy()
        if n == -1:
            return A
        n = -n
    else:
        if n == 1:
            return a.copy()
        A = a
    if n < 4:
        if n == 2:
            return np.dot(A, A)
        if n == 3:
            return np.dot(np.dot(A, A), A)
    else:
        acc = A
        exp = n
        ret = acc
        flag = True
        while exp != 0:
            if exp & 1:
                if flag:
                    ret = acc
                    flag = False
                else:
                    ret = np.dot(ret, acc)
            acc = np.dot(acc, acc)
            exp = exp >> 1
        return ret