import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_no_copy(self):
    array = np.arange(100, dtype=np.float32)
    cuda.to_device(array, copy=False)