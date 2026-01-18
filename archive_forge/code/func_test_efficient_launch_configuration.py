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
def test_efficient_launch_configuration(self):

    @guvectorize(['void(float32[:], float32[:], float32[:])'], '(n),(n)->(n)', nopython=True, target='cuda')
    def numba_dist_cuda2(a, b, dist):
        len = a.shape[0]
        for i in range(len):
            dist[i] = a[i] * b[i]
    a = np.random.rand(524288 * 2).astype('float32').reshape((524288, 2))
    b = np.random.rand(524288 * 2).astype('float32').reshape((524288, 2))
    dist = np.zeros_like(a)
    with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
        with warnings.catch_warnings(record=True) as w:
            numba_dist_cuda2(a, b, dist)
            self.assertEqual(len(w), 0)