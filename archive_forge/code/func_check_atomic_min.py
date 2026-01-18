import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def check_atomic_min(self, dtype, lo, hi):
    vals = np.random.randint(lo, hi, size=(32, 32)).astype(dtype)
    res = np.array([65535], dtype=vals.dtype)
    cuda_func = cuda.jit(atomic_min)
    cuda_func[32, 32](res, vals)
    gold = np.min(vals)
    np.testing.assert_equal(res, gold)