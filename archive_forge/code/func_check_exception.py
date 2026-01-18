from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_exception(self, pyfunc, msg):
    cfunc = jit(nopython=True)(pyfunc)
    with self.assertRaises(BaseException):
        pyfunc(self.zero_size)
    with self.assertRaises(ValueError) as e:
        cfunc(self.zero_size)
    self.assertIn(msg, str(e.exception))