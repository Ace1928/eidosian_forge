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
def test_copy_unspecified_return(self):

    @guvectorize([(float32[:], float32[:])], '(x)->(x)', target='cuda')
    def copy(A, B):
        for i in range(B.size):
            B[i] = A[i]
    A = np.arange(10, dtype=np.float32) + 1
    B = np.zeros_like(A)
    copy(A, out=B)
    self.assertTrue(np.allclose(A, B))