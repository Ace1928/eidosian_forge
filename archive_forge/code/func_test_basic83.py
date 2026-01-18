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
def test_basic83(self):
    """ neighborhood, with standard_indexing and cval"""

    def kernel(a, b):
        cumul = 0
        for i in range(-3, 1):
            for j in range(-3, 1):
                cumul += a[i, j] + b[1, 3]
        return cumul / 9.0
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    b = a.copy()

    def __kernel(a, b, neighborhood):
        self.check_stencil_arrays(a, b, neighborhood=neighborhood)
        __retdtype = kernel(a, b)
        __b0 = np.full(a.shape, 1.5, dtype=type(__retdtype))
        for __bn in range(3, a.shape[1]):
            for __an in range(3, a.shape[0]):
                cumul = 0
                for i in range(-3, 1):
                    for j in range(-3, 1):
                        cumul += a[__an + i, __bn + j] + b[1, 3]
                __b0[__an, __bn] = cumul / 9.0
        return __b0
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    b = a.copy()
    nh = ((-3, 0), (-3, 0))
    expected = __kernel(a, b, nh)
    self.check_against_expected(kernel, expected, a, b, options={'neighborhood': nh, 'standard_indexing': 'b', 'cval': 1.5})