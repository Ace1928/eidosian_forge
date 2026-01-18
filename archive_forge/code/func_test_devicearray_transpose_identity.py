import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_transpose_identity(self):
    original = np.array(np.arange(24)).reshape(3, 4, 2)
    array = np.transpose(cuda.to_device(original), axes=(0, 1, 2)).copy_to_host()
    self.assertTrue(np.all(array == original))