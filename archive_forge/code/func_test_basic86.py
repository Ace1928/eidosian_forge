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
def test_basic86(self):
    """ bad kwarg """

    def kernel(a):
        return a[0, 0]
    a = np.arange(10.0 * 20.0).reshape(10, 20)
    self.check_exceptions(kernel, a, options={'bad': 10}, expected_exception=[ValueError, TypingError])