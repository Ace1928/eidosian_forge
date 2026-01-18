import gc
import numpy as np
import unittest
from numba import njit
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin
def test_del_at_beginning_of_loop(self):
    """
        Test issue #1734
        """

    @njit
    def f(arr):
        res = 0
        for i in (0, 1):
            t = arr[i]
            if t[i] > 1:
                res += t[i]
        return res
    arr = np.ones((2, 2))
    init_stats = rtsys.get_allocation_stats()
    f(arr)
    cur_stats = rtsys.get_allocation_stats()
    self.assertEqual(cur_stats.alloc - init_stats.alloc, cur_stats.free - init_stats.free)