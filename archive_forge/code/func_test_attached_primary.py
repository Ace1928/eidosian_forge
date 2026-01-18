import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
def test_attached_primary(self, extra_work=lambda: None):
    the_driver = driver.driver
    if driver.USE_NV_BINDING:
        dev = driver.binding.CUdevice(0)
        hctx = the_driver.cuDevicePrimaryCtxRetain(dev)
    else:
        dev = 0
        hctx = driver.drvapi.cu_context()
        the_driver.cuDevicePrimaryCtxRetain(byref(hctx), dev)
    try:
        ctx = driver.Context(weakref.proxy(self), hctx)
        ctx.push()
        my_ctx = cuda.current_context()
        if driver.USE_NV_BINDING:
            self.assertEqual(int(my_ctx.handle), int(ctx.handle))
        else:
            self.assertEqual(my_ctx.handle.value, ctx.handle.value)
        extra_work()
    finally:
        ctx.pop()
        the_driver.cuDevicePrimaryCtxRelease(dev)