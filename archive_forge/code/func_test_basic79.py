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
def test_basic79(self):
    """ neighborhood, two incompatible args """

    def kernel(a, b):
        cumul = 0
        for i in range(-3, 1):
            for j in range(-3, 1):
                cumul += a[i, j] + b[i, j]
        return cumul / 9.0
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    b = np.arange(10.0 * 20.0).reshape(10, 10, 2)
    ex = self.exception_dict(stencil=TypingError, parfor=TypingError, njit=TypingError)
    self.check_exceptions(kernel, a, b, options={'neighborhood': ((-3, 0), (-3, 0))}, expected_exception=ex)