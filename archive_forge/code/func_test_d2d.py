import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def test_d2d(self):
    hst = np.arange(100, dtype=np.uint32)
    hst2 = np.empty_like(hst)
    sz = hst.size * hst.dtype.itemsize
    dev1 = self.context.memalloc(sz)
    dev2 = self.context.memalloc(sz)
    driver.host_to_device(dev1, hst, sz)
    driver.device_to_device(dev2, dev1, sz)
    driver.device_to_host(hst2, dev2, sz)
    self.assertTrue(np.all(hst == hst2))