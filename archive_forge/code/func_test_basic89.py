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
def test_basic89(self):
    """ basic multiple return"""

    def kernel(a):
        if a[0, 1] > 10:
            return 10.0
        elif a[0, 3] < 8:
            return a[0, 0]
        else:
            return 7.0

    def __kernel(a, neighborhood):
        self.check_stencil_arrays(a, neighborhood=neighborhood)
        __retdtype = kernel(a)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __b in range(0, a.shape[1] - 3):
            for __a in range(0, a.shape[0]):
                if a[__a + 0, __b + 1] > 10:
                    __b0[__a, __b] = 10.0
                elif a[__a + 0, __b + 3] < 8:
                    __b0[__a, __b] = a[__a + 0, __b + 0]
                else:
                    __b0[__a, __b] = 7.0
        return __b0
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    expected = __kernel(a, None)
    self.check_against_expected(kernel, expected, a)