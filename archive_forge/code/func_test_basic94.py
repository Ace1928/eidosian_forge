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
def test_basic94(self):
    """ Issue #3528. Support for slices. """

    def kernel(a):
        return np.median(a[-1:2, -1:2])

    def __kernel(a, neighborhood):
        self.check_stencil_arrays(a, neighborhood=neighborhood)
        __retdtype = kernel(a)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __bn in range(1, a.shape[1] - 1):
            for __an in range(1, a.shape[0] - 1):
                __b0[__an, __bn] = np.median(a[__an + -1:__an + 2, __bn + -1:__bn + 2])
        return __b0
    a = np.arange(20, dtype=np.uint32).reshape(4, 5)
    nh = ((-1, 1), (-1, 1))
    expected = __kernel(a, nh)
    self.check_against_expected(kernel, expected, a, options={'neighborhood': nh})