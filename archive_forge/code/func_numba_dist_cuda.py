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
@guvectorize(['void(float32[:], float32[:], float32[:])'], '(n),(n)->(n)', target='cuda')
def numba_dist_cuda(a, b, dist):
    len = a.shape[0]
    for i in range(len):
        dist[i] = a[i] * b[i]