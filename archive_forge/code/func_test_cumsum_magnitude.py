from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_cumsum_magnitude(self):
    self.check_aggregation_magnitude(array_cumsum)
    self.check_aggregation_magnitude(array_cumsum_global)