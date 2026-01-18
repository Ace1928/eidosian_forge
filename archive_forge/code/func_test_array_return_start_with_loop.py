import numpy as np
from numba import typeof, njit
from numba.tests.support import MemoryLeakMixin
import unittest
def test_array_return_start_with_loop(self):
    """
        A bug breaks array return if the function starts with a loop
        """
    a = np.arange(10)
    at = typeof(a)
    cfunc = njit((at,))(array_return_start_with_loop)
    self.assertIs(a, cfunc(a))