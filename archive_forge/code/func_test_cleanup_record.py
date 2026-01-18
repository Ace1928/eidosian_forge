import gc
import itertools
import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase
from numba.np import numpy_support
def test_cleanup_record(self):
    dtype = np.dtype([('x', np.float64), ('y', np.float64)])
    recarr = np.zeros(1, dtype=dtype)
    self.check_argument_cleanup(numpy_support.from_dtype(dtype), recarr[0])