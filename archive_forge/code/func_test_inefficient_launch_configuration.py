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
def test_inefficient_launch_configuration(self):

    @guvectorize(['void(float32[:], float32[:], float32[:])'], '(n),(n)->(n)', target='cuda')
    def numba_dist_cuda(a, b, dist):
        len = a.shape[0]
        for i in range(len):
            dist[i] = a[i] * b[i]
    a = np.random.rand(1024 * 32).astype('float32')
    b = np.random.rand(1024 * 32).astype('float32')
    dist = np.zeros(a.shape[0]).astype('float32')
    with override_config('CUDA_LOW_OCCUPANCY_WARNINGS', 1):
        with warnings.catch_warnings(record=True) as w:
            numba_dist_cuda(a, b, dist)
            self.assertEqual(w[0].category, NumbaPerformanceWarning)
            self.assertIn('Grid size', str(w[0].message))
            self.assertIn('low occupancy', str(w[0].message))