import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
def test_basic76(self):
    """neighborhood, mixed range span"""

    def kernel(a):
        cumul = 0
        zz = 12.0
        for i in range(-3, 0):
            t = zz + i
            for j in range(-3, 4):
                cumul += a[i, j] + t
        return cumul / (10 * 5)

    def __kernel(a, neighborhood):
        self.check_stencil_arrays(a, neighborhood=neighborhood)
        __retdtype = kernel(a)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __bn in range(3, a.shape[1] - 3):
            for __an in range(3, a.shape[0]):
                cumul = 0
                zz = 12.0
                for i in range(-3, 0):
                    t = zz + i
                    for j in range(-3, 4):
                        cumul += a[__an + i, __bn + j] + t
                __b0[__an, __bn] = cumul / 50
        return __b0
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    nh = ((-3, -1), (-3, 3))
    expected = __kernel(a, nh)
    self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})