import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_nanmin_double_oneindex(self):
    vals = np.random.randint(0, 128, size=32).astype(np.float64)
    vals[1::2] = np.nan
    res = np.array([128], np.float64)
    cuda_func = cuda.jit('void(float64[:], float64[:])')(atomic_min_double_oneindex)
    cuda_func[1, 32](res, vals)
    gold = np.nanmin(vals)
    np.testing.assert_equal(res, gold)