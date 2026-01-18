import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_len_2d(self):
    ary = np.empty((3, 5))
    dary = cuda.device_array((3, 5))
    self.assertEqual(len(ary), len(dary))