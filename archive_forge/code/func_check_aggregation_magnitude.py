from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_aggregation_magnitude(self, pyfunc, is_prod=False):
    """
        Check that integer overflows are avoided (issue #931).
        """
    n_items = 2 if is_prod else 10
    arr = (np.arange(n_items) + 40000).astype('int16')
    npr, nbr = run_comparative(pyfunc, arr)
    self.assertPreciseEqual(npr, nbr)
    arr = (np.arange(10) + 2 ** 60).astype('int64')
    npr, nbr = run_comparative(pyfunc, arr)
    self.assertPreciseEqual(npr, nbr)
    arr = arr.astype('uint64')
    npr, nbr = run_comparative(pyfunc, arr)
    self.assertPreciseEqual(npr, nbr)