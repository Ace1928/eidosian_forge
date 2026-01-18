from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_max_npdatetime(self):
    self.check_npdatetime(array_max)
    self.check_nptimedelta(array_max)