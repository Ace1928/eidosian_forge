import numpy as np
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba import cuda, float64
import unittest
def test_device_opt(self):
    sig = (float64, float64, float64)
    device = cuda.jit(sig, device=True)(device_func)
    ptx = device.inspect_asm(sig)
    self.assertIn('fma.rn.f64', ptx)