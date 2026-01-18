import itertools
import numpy as np
from numba.cuda.cudadrv import devicearray
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase
from numba.cuda.testing import skip_on_cudasim
def test_devicearray_view_bad_not_c_contig(self):
    original = np.array(np.arange(32), dtype='i2').reshape(4, 8)
    array = cuda.to_device(original)[:, ::2]
    with self.assertRaises(ValueError) as e:
        array.view('i4')
    msg = str(e.exception)
    self.assertIn('To change to a dtype of a different size,', msg)
    contiguous_pre_np123 = 'the array must be C-contiguous' in msg
    contiguous_post_np123 = 'the last axis must be contiguous' in msg
    self.assertTrue(contiguous_pre_np123 or contiguous_post_np123, 'Expected message to mention contiguity')