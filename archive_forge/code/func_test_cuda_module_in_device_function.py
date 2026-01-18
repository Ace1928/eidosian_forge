import threading
import numpy as np
from numba import cuda
from numba.cuda.testing import CUDATestCase, skip_unless_cudasim
import numba.cuda.simulator as simulator
import unittest
def test_cuda_module_in_device_function(self):
    """
        Discovered in https://github.com/numba/numba/issues/1837.
        When the `cuda` module is referenced in a device function,
        it does not have the kernel API (e.g. cuda.threadIdx, cuda.shared)
        """
    from numba.cuda.tests.cudasim import support
    inner = support.cuda_module_in_device_function

    @cuda.jit
    def outer(out):
        tid = inner()
        if tid < out.size:
            out[tid] = tid
    arr = np.zeros(10, dtype=np.int32)
    outer[1, 11](arr)
    expected = np.arange(arr.size, dtype=np.int32)
    np.testing.assert_equal(expected, arr)