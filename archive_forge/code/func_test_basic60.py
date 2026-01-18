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
def test_basic60(self):
    """3 args, mix of array, relative and standard indexing,
        tuple pass through"""

    def kernel(a, b, c):
        return a[0, 1] + b[1, 1] + c[0]
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(12.0).reshape(3, 4)
    c = (10,)
    ex = self.exception_dict(parfor=ValueError)
    self.check_exceptions(kernel, a, b, c, options={'standard_indexing': ['b', 'c']}, expected_exception=ex)