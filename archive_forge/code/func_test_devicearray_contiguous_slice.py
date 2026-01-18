import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_contiguous_slice(self):
    a = np.arange(25).reshape(5, 5, order='F')
    s = np.full(fill_value=5, shape=(5,))
    d = cuda.to_device(a)
    a[2] = s
    with self.assertRaises(ValueError) as e:
        d[2].copy_to_device(s)
    self.assertEqual(devicearray.errmsg_contiguous_buffer, str(e.exception))