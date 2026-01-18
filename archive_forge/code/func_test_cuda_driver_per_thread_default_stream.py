from ctypes import byref, c_int, c_void_p, sizeof
from numba.cuda.cudadrv.driver import (host_to_device, device_to_host, driver,
from numba.cuda.cudadrv import devices, drvapi, driver as _driver
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_cuda_driver_per_thread_default_stream(self):
    ds = self.context.get_per_thread_default_stream()
    self.assertIn('Per-thread default CUDA stream', repr(ds))
    self.assertEqual(2, int(ds))
    self.assertTrue(ds)
    self.assertFalse(ds.external)