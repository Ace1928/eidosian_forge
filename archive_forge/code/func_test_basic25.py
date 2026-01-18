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
def test_basic25(self):
    """no idx on 2d arr"""
    a = np.arange(12).reshape(3, 4)

    def kernel(a):
        return 1.0
    self.check_exceptions(kernel, a, expected_exception=[ValueError, NumbaValueError])