import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_device_array_like_2d_transpose(self):
    d_a = cuda.device_array((10, 12), order='C')
    self._test_against_array_core(d_a.T)