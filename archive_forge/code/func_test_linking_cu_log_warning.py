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
def test_linking_cu_log_warning(self):
    bar = cuda.declare_device('bar', 'int32(int32)')
    link = str(test_data_dir / 'warn.cu')
    with warnings.catch_warnings(record=True) as w:
        ignore_internal_warnings()

        @cuda.jit('void(int32)', link=[link])
        def kernel(x):
            bar(x)
    self.assertEqual(len(w), 1, 'Expected warnings from NVRTC')
    self.assertIn('NVRTC log messages', str(w[0].message))
    self.assertIn('declared but never referenced', str(w[0].message))