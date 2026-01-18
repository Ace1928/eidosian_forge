import gc
import numpy as np
import unittest
from numba import njit
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin
def test_no_return(self):
    """
        Test issue #1291
        """

    @njit
    def foo(n):
        for i in range(n):
            temp = np.zeros(2)
        return 0
    n = 10
    init_stats = rtsys.get_allocation_stats()
    foo(n)
    cur_stats = rtsys.get_allocation_stats()
    self.assertEqual(cur_stats.alloc - init_stats.alloc, n)
    self.assertEqual(cur_stats.free - init_stats.free, n)