import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_masked_array(self):
    h_arr = np.random.random(10)
    h_mask = np.random.randint(2, size=10, dtype='bool')
    c_arr = cuda.to_device(h_arr)
    c_mask = cuda.to_device(h_mask)
    masked_cuda_array_interface = c_arr.__cuda_array_interface__.copy()
    masked_cuda_array_interface['mask'] = c_mask
    with self.assertRaises(NotImplementedError) as raises:
        cuda.from_cuda_array_interface(masked_cuda_array_interface)
    expected_msg = 'Masked arrays are not supported'
    self.assertIn(expected_msg, str(raises.exception))