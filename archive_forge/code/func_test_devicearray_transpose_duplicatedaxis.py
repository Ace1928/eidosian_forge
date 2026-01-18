import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_transpose_duplicatedaxis(self):
    gpumem = cuda.to_device(np.array(np.arange(12)).reshape(3, 4))
    with self.assertRaises(ValueError) as e:
        np.transpose(gpumem, axes=(0, 0))
    self.assertIn(str(e.exception), container=['invalid axes list (0, 0)', 'repeated axis in transpose'])