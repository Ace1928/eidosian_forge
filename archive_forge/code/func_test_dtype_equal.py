import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def test_dtype_equal(self):
    pyfunc = dtype_eq_int64
    self.check_unary(pyfunc, np.empty(1, dtype=np.int16))
    self.check_unary(pyfunc, np.empty(1, dtype=np.int64))