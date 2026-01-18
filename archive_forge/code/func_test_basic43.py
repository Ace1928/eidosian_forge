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
def test_basic43(self):
    """2 args more complexity"""

    def kernel(a, b):
        return a[0, 1] + a[1, 2] + b[-2, 0] + b[0, -1]

    def __kernel(a, b, neighborhood):
        self.check_stencil_arrays(a, b, neighborhood=neighborhood)
        __retdtype = kernel(a, b)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __b in range(1, a.shape[1] - 2):
            for __a in range(2, a.shape[0] - 1):
                __b0[__a, __b] = a[__a + 0, __b + 1] + a[__a + 1, __b + 2] + b[__a + -2, __b + 0] + b[__a + 0, __b + -1]
        return __b0
    a = np.arange(30.0).reshape(5, 6)
    b = np.arange(30.0).reshape(5, 6)
    expected = __kernel(a, b, None)
    self.check_against_expected(kernel, expected, a, b)