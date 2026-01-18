import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_view_ok_not_c_contig(self):
    original = np.array(np.arange(32), dtype='i2').reshape(4, 8)
    array = cuda.to_device(original)[:, ::2]
    original = original[:, ::2]
    np.testing.assert_array_equal(array.view('u2').copy_to_host(), original.view('u2'))