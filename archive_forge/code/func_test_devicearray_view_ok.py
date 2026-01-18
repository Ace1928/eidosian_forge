import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_view_ok(self):
    original = np.array(np.arange(12), dtype='i2').reshape(3, 4)
    array = cuda.to_device(original)
    for dtype in ('i4', 'u4', 'i8', 'f8'):
        with self.subTest(dtype=dtype):
            np.testing.assert_array_equal(array.view(dtype).copy_to_host(), original.view(dtype))