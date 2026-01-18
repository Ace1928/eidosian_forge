import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_as_cuda_array(self):
    h_arr = np.arange(10)
    self.assertFalse(cuda.is_cuda_array(h_arr))
    d_arr = cuda.to_device(h_arr)
    self.assertTrue(cuda.is_cuda_array(d_arr))
    my_arr = ForeignArray(d_arr)
    self.assertTrue(cuda.is_cuda_array(my_arr))
    wrapped = cuda.as_cuda_array(my_arr)
    self.assertTrue(cuda.is_cuda_array(wrapped))
    np.testing.assert_array_equal(wrapped.copy_to_host(), h_arr)
    np.testing.assert_array_equal(d_arr.copy_to_host(), h_arr)
    self.assertPointersEqual(wrapped, d_arr)