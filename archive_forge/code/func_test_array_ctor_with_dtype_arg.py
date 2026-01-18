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
def test_array_ctor_with_dtype_arg(self):
    pyfunc = array_ctor
    cfunc = jit(nopython=True)(pyfunc)
    n = 2
    args = (n, np.int32)
    np.testing.assert_array_equal(pyfunc(*args), cfunc(*args))
    args = (n, np.dtype('int32'))
    np.testing.assert_array_equal(pyfunc(*args), cfunc(*args))
    args = (n, np.float32)
    np.testing.assert_array_equal(pyfunc(*args), cfunc(*args))
    args = (n, np.dtype('f4'))
    np.testing.assert_array_equal(pyfunc(*args), cfunc(*args))