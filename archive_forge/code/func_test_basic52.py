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
def test_basic52(self):
    """3 args, standard_indexing on middle arg """

    def kernel(a, b, c):
        return a[0, 1] + b[0, 1] + c[1, 2]

    def __kernel(a, b, c, neighborhood):
        self.check_stencil_arrays(a, c, neighborhood=neighborhood)
        __retdtype = kernel(a, b, c)
        __b0 = np.full(a.shape, 0, dtype=type(__retdtype))
        for __b in range(0, a.shape[1] - 2):
            for __a in range(0, a.shape[0] - 1):
                __b0[__a, __b] = a[__a + 0, __b + 1] + b[0, 1] + c[__a + 1, __b + 2]
        return __b0
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(4.0).reshape(2, 2)
    c = np.arange(12.0).reshape(3, 4)
    expected = __kernel(a, b, c, None)
    self.check_against_expected(kernel, expected, a, b, c, options={'standard_indexing': 'b'})