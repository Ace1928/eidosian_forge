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
def test_basic39(self):
    """cval is func expr"""

    def kernel(a):
        return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
    cval = np.sin(3.0) + np.cos(2)

    def __kernel(a, neighborhood):
        self.check_stencil_arrays(a, neighborhood=neighborhood)
        __retdtype = kernel(a)
        __b0 = np.full(a.shape, cval, dtype=type(__retdtype))
        for __b in range(1, a.shape[1] - 1):
            for __a in range(0, a.shape[0] - 1):
                __b0[__a, __b] = a[__a + 0, __b + 1] + a[__a + 0, __b + -1] + a[__a + 1, __b + -1] + a[__a + 1, __b + -1]
        return __b0
    a = np.arange(12.0).reshape(3, 4)
    expected = __kernel(a, None)
    self.check_against_expected(kernel, expected, a, options={'cval': cval})