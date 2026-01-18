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
def test_basic87(self):
    """ reserved arg name in use """

    def kernel(__sentinel__):
        return __sentinel__[0, 0]

    def __kernel(__sentinel__, neighborhood):
        self.check_stencil_arrays(__sentinel__, neighborhood=neighborhood)
        __retdtype = kernel(__sentinel__)
        __b0 = np.full(__sentinel__.shape, 0, dtype=type(__retdtype))
        for __b in range(0, __sentinel__.shape[1]):
            for __a in range(0, __sentinel__.shape[0]):
                __b0[__a, __b] = __sentinel__[__a + 0, __b + 0]
        return __b0
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    expected = __kernel(a, None)
    self.check_against_expected(kernel, expected, a)