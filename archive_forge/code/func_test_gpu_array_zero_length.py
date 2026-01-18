import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim, skip_unless_cudasim
from numba import config, cuda
def test_gpu_array_zero_length(self):
    x = np.arange(0)
    dx = cuda.to_device(x)
    hx = dx.copy_to_host()
    self.assertEqual(x.shape, dx.shape)
    self.assertEqual(x.size, dx.size)
    self.assertEqual(x.shape, hx.shape)
    self.assertEqual(x.size, hx.size)