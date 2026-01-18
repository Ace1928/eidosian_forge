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
def check_tuple_arg(self, a, b):

    @guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->()', target='cuda')
    def gu_reduce(x, y, r):
        s = 0
        for i in range(len(x)):
            s += x[i] * y[i]
        r[0] = s
    r = gu_reduce(a, b)
    expected = np.sum(np.asarray(a) * np.asarray(b), axis=1)
    np.testing.assert_equal(expected, r)