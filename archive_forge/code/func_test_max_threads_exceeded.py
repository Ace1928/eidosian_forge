import numpy as np
import warnings
from numba.cuda.testing import unittest
from numba.cuda.testing import (skip_on_cudasim, skip_if_cuda_includes_missing)
from numba.cuda.testing import CUDATestCase, test_data_dir
from numba.cuda.cudadrv.driver import (CudaAPIError, Linker,
from numba.cuda.cudadrv.error import NvrtcError
from numba.cuda import require_context
from numba.tests.support import ignore_internal_warnings
from numba import cuda, void, float64, int64, int32, typeof, float32
def test_max_threads_exceeded(self):
    compiled = cuda.jit('void(int32[::1])')(simple_maxthreads)
    max_threads = compiled.get_max_threads_per_block()
    nelem = max_threads + 1
    ary = np.empty(nelem, dtype=np.int32)
    try:
        compiled[1, nelem](ary)
    except CudaAPIError as e:
        self.assertIn('cuLaunchKernel', e.msg)