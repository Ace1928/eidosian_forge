from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def convert_to_float_and_check(a, q, abs_tol=1e-14):
    expected = pyfunc(a, q).astype(np.float64)
    got = cfunc(a, q)
    self.assertPreciseEqual(got, expected, abs_tol=abs_tol)