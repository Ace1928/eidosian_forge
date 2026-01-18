import traceback
import threading
import multiprocessing
import numpy as np
from numba import cuda
from numba.cuda.testing import (skip_on_cudasim, skip_under_cuda_memcheck,
import unittest
def test_invalid_context_error_with_d2h(self):

    def d2h(arr, out):
        out[:] = arr.copy_to_host()
    arr = np.arange(1, 4)
    out = np.zeros_like(arr)
    darr = cuda.to_device(arr)
    th = threading.Thread(target=d2h, args=[darr, out])
    th.start()
    th.join()
    np.testing.assert_equal(arr, out)