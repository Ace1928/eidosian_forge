import traceback
import threading
import multiprocessing
import numpy as np
from numba import cuda
from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
import unittest
def test_invalid_context_error_with_d2d(self):

    def d2d(dst, src):
        dst.copy_to_device(src)
    arr = np.arange(100)
    common = cuda.to_device(arr)
    darr = cuda.to_device(np.zeros(common.shape, dtype=common.dtype))
    th = threading.Thread(target=d2d, args=[darr, common])
    th.start()
    th.join()
    np.testing.assert_equal(darr.copy_to_host(), arr)