import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_array_views(self):
    """Views created via array interface support:
            - Strided slices
            - Strided slices
        """
    h_arr = np.random.random(10)
    c_arr = cuda.to_device(h_arr)
    arr = cuda.as_cuda_array(c_arr)
    np.testing.assert_array_equal(arr.copy_to_host(), h_arr)
    np.testing.assert_array_equal(arr[:].copy_to_host(), h_arr)
    np.testing.assert_array_equal(arr[:5].copy_to_host(), h_arr[:5])
    np.testing.assert_array_equal(arr[::2].copy_to_host(), h_arr[::2])
    arr_strided = cuda.as_cuda_array(c_arr[::2])
    np.testing.assert_array_equal(arr_strided.copy_to_host(), h_arr[::2])
    self.assertEqual(arr[::2].shape, arr_strided.shape)
    self.assertEqual(arr[::2].strides, arr_strided.strides)
    self.assertEqual(arr[::2].dtype.itemsize, arr_strided.dtype.itemsize)
    self.assertEqual(arr[::2].alloc_size, arr_strided.alloc_size)
    self.assertEqual(arr[::2].nbytes, arr_strided.size * arr_strided.dtype.itemsize)
    arr[:5] = np.pi
    np.testing.assert_array_equal(c_arr.copy_to_host(), np.concatenate((np.full(5, np.pi), h_arr[5:])))
    arr[:5] = arr[5:]
    np.testing.assert_array_equal(c_arr.copy_to_host(), np.concatenate((h_arr[5:], h_arr[5:])))
    arr[:] = cuda.to_device(h_arr)
    np.testing.assert_array_equal(c_arr.copy_to_host(), h_arr)
    arr[::2] = np.pi
    np.testing.assert_array_equal(c_arr.copy_to_host()[::2], np.full(5, np.pi))
    np.testing.assert_array_equal(c_arr.copy_to_host()[1::2], h_arr[1::2])