from ctypes import byref, c_int, c_void_p, sizeof
from numba.cuda.cudadrv.driver import (host_to_device, device_to_host, driver,
from numba.cuda.cudadrv import devices, drvapi, driver as _driver
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_cuda_driver_external_stream(self):
    if _driver.USE_NV_BINDING:
        handle = driver.cuStreamCreate(0)
        ptr = int(handle)
    else:
        handle = drvapi.cu_stream()
        driver.cuStreamCreate(byref(handle), 0)
        ptr = handle.value
    s = self.context.create_external_stream(ptr)
    self.assertIn('External CUDA stream', repr(s))
    self.assertNotIn('efault', repr(s))
    self.assertEqual(ptr, int(s))
    self.assertTrue(s)
    self.assertTrue(s.external)