import ctypes
import numpy as np
from numba.cuda.cudadrv import driver, drvapi, devices
from numba.cuda.testing import unittest, ContextResettingTestCase
from numba.cuda.testing import skip_on_cudasim
def test_memcpy(self):
    hstary = np.arange(100, dtype=np.uint32)
    hstary2 = np.arange(100, dtype=np.uint32)
    sz = hstary.size * hstary.dtype.itemsize
    devary = self.context.memalloc(sz)
    driver.host_to_device(devary, hstary, sz)
    driver.device_to_host(hstary2, devary, sz)
    self.assertTrue(np.all(hstary == hstary2))