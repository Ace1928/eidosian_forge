import gc
import numpy as np
import unittest
from numba import njit
from numba.core.runtime import rtsys
from numba.tests.support import TestCase, EnableNRTStatsMixin
def test_invalid_computation_of_lifetime(self):
    """
        Test issue #1573
        """

    @njit
    def if_with_allocation_and_initialization(arr1, test1):
        tmp_arr = np.zeros_like(arr1)
        for i in range(tmp_arr.shape[0]):
            pass
        if test1:
            np.zeros_like(arr1)
        return tmp_arr
    arr = np.random.random((5, 5))
    init_stats = rtsys.get_allocation_stats()
    if_with_allocation_and_initialization(arr, False)
    cur_stats = rtsys.get_allocation_stats()
    self.assertEqual(cur_stats.alloc - init_stats.alloc, cur_stats.free - init_stats.free)