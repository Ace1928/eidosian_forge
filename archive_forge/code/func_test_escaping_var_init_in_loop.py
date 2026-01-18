import gc
import numpy as np
import unittest
from numba import njit
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin
def test_escaping_var_init_in_loop(self):
    """
        Test issue #1297
        """

    @njit
    def g(n):
        x = np.zeros((n, 2))
        for i in range(n):
            y = x[i]
        for i in range(n):
            y = x[i]
        return 0
    init_stats = rtsys.get_allocation_stats()
    g(10)
    cur_stats = rtsys.get_allocation_stats()
    self.assertEqual(cur_stats.alloc - init_stats.alloc, 1)
    self.assertEqual(cur_stats.free - init_stats.free, 1)