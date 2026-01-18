import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
def test_memalloc(self):
    mgr = cuda.current_context().memory_manager
    arr_1 = np.arange(10)
    d_arr_1 = cuda.device_array_like(arr_1)
    self.assertTrue(mgr.memalloc_called)
    self.assertEqual(mgr.count, 1)
    self.assertEqual(mgr.allocations[1], arr_1.nbytes)
    arr_2 = np.arange(5)
    d_arr_2 = cuda.device_array_like(arr_2)
    self.assertEqual(mgr.count, 2)
    self.assertEqual(mgr.allocations[2], arr_2.nbytes)
    del d_arr_1
    self.assertNotIn(1, mgr.allocations)
    self.assertIn(2, mgr.allocations)
    del d_arr_2
    self.assertNotIn(2, mgr.allocations)