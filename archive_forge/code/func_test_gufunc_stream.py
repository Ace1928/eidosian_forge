import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
def test_gufunc_stream(self):
    gufunc = _get_matmulcore_gufunc()
    matrix_ct = 1001
    A = np.arange(matrix_ct * 2 * 4, dtype=np.float32).reshape(matrix_ct, 2, 4)
    B = np.arange(matrix_ct * 4 * 5, dtype=np.float32).reshape(matrix_ct, 4, 5)
    stream = cuda.stream()
    dA = cuda.to_device(A, stream)
    dB = cuda.to_device(B, stream)
    dC = cuda.device_array(shape=(1001, 2, 5), dtype=A.dtype, stream=stream)
    dC = gufunc(dA, dB, out=dC, stream=stream)
    C = dC.copy_to_host(stream=stream)
    stream.synchronize()
    Gold = np.matmul(A, B)
    self.assertTrue(np.allclose(C, Gold))