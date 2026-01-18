import numpy as np
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
def test_assign_empty_slice(self):
    N = 0
    a = range(N)
    arr = cuda.device_array(len(a))
    arr[:] = cuda.to_device(a)