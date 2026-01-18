import multiprocessing as mp
import os
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError, driver
from numba.cuda.cudadrv.error import CudaSupportError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
def test_init_failure_raising(self):
    expected = 'Error at driver init: CUDA_ERROR_UNKNOWN (999)'
    self._test_init_failure(cuInit_raising_test, expected)