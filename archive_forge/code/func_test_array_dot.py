from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
@needs_blas
def test_array_dot(self):
    pyfunc = array_dot
    cfunc = jit(nopython=True)(pyfunc)
    a = np.arange(20.0).reshape(4, 5)
    b = np.arange(5.0)
    np.testing.assert_equal(pyfunc(a, b), cfunc(a, b))
    pyfunc = array_dot_chain
    cfunc = jit(nopython=True)(pyfunc)
    a = np.arange(16.0).reshape(4, 4)
    np.testing.assert_equal(pyfunc(a, a), cfunc(a, a))