import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_view_bad_itemsize(self):
    original = np.array(np.arange(12), dtype='i2').reshape(4, 3)
    array = cuda.to_device(original)
    with self.assertRaises(ValueError) as e:
        array.view('i4')
    self.assertEqual('When changing to a larger dtype, its size must be a divisor of the total size in bytes of the last axis of the array.', str(e.exception))