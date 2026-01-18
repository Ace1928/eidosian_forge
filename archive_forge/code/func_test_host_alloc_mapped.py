import numpy as np
from numba.cuda.cudadrv import driver
from numba import cuda
from numba.cuda.testing import unittest, ContextResettingTestCase
def test_host_alloc_mapped(self):
    ary = cuda.mapped_array(10, dtype=np.uint32)
    ary.fill(123)
    self.assertTrue(all(ary == 123))
    driver.device_memset(ary, 0, driver.device_memory_size(ary))
    self.assertTrue(all(ary == 0))
    self.assertTrue(sum(ary != 0) == 0)