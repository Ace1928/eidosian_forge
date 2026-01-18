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
def test_basic42(self):
    """2 args! rel arrays very close in size"""

    def kernel(a, b):
        return a[0, 1] + b[0, -2]
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(9.0).reshape(3, 3)
    self.check_exceptions(kernel, a, b, expected_exception=[ValueError, AssertionError])