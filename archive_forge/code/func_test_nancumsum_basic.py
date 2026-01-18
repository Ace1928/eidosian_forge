from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_nancumsum_basic(self):
    self.check_cumulative(array_nancumsum)
    self.check_nan_cumulative(array_nancumsum)