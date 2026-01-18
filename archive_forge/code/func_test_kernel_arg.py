import numpy as np
from numba import vectorize, guvectorize
from numba import cuda
from numba.cuda.cudadrv import driver
from numba.cuda.testing import unittest, ContextResettingTestCase, ForeignArray
from numba.cuda.testing import skip_on_cudasim, skip_if_external_memmgr
from numba.tests.support import linux_only, override_config
from unittest.mock import call, patch
def test_kernel_arg(self):
    h_arr = np.arange(10)
    d_arr = cuda.to_device(h_arr)
    my_arr = ForeignArray(d_arr)
    wrapped = cuda.as_cuda_array(my_arr)

    @cuda.jit
    def mutate(arr, val):
        i = cuda.grid(1)
        if i >= len(arr):
            return
        arr[i] += val
    val = 7
    mutate.forall(wrapped.size)(wrapped, val)
    np.testing.assert_array_equal(wrapped.copy_to_host(), h_arr + val)
    np.testing.assert_array_equal(d_arr.copy_to_host(), h_arr + val)