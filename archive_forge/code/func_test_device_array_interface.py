import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_device_array_interface(self):
    dary = cuda.device_array(shape=100)
    devicearray.verify_cuda_ndarray_interface(dary)
    ary = np.empty(100)
    dary = cuda.to_device(ary)
    devicearray.verify_cuda_ndarray_interface(dary)
    ary = np.asarray(1.234)
    dary = cuda.to_device(ary)
    self.assertEqual(dary.ndim, 0)
    devicearray.verify_cuda_ndarray_interface(dary)