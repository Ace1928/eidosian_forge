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
def test_basic38(self):
    """cval is complex"""

    def kernel(a):
        return a[0, 1] + a[0, -1] + a[1, -1] + a[1, -1]
    a = np.arange(12.0).reshape(3, 4)
    ex = self.exception_dict(stencil=NumbaValueError, parfor=ValueError, njit=NumbaValueError)
    self.check_exceptions(kernel, a, options={'cval': 1j}, expected_exception=ex)