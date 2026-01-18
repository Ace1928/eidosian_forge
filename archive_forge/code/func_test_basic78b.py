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
def test_basic78b(self):
    """ neighborhood, two args, -ve range, +ve range """

    def kernel(a, b):
        cumul = 0
        for i in range(-6, -2):
            for j in range(2, 10):
                cumul += a[i, j] + b[i, j]
        return cumul / 9.0

    def __kernel(a, b, neighborhood):
        self.check_stencil_arrays(a, b, neighborhood=neighborhood)
        __retdtype = kernel(a, b)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __bn in range(0, a.shape[1] - 9):
            for __an in range(6, a.shape[0]):
                cumul = 0
                for i in range(-6, -2):
                    for j in range(2, 10):
                        cumul += a[__an + i, __bn + j] + b[__an + i, __bn + j]
                __b0[__an, __bn] = cumul / 9.0
        return __b0
    a = np.arange(15.0 * 20.0).reshape(15, 20)
    b = np.arange(15.0 * 20.0).reshape(15, 20)
    nh = ((-6, -3), (2, 9))
    expected = __kernel(a, b, nh)
    self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh})