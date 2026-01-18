from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_nanmedian_basic(self):
    pyfunc = array_nanmedian_global
    self.check_median_basic(pyfunc, self._array_variations)