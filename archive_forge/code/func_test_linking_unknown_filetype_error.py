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
def test_linking_unknown_filetype_error(self):
    expected_err = "Don't know how to link file with extension .cuh"
    with self.assertRaisesRegex(RuntimeError, expected_err):

        @cuda.jit('void()', link=['header.cuh'])
        def kernel():
            pass