import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_add_double(self):
    idx = np.random.randint(0, 32, size=32, dtype=np.int64)
    ary = np.zeros(32, np.float64)
    ary_wrap = ary.copy()
    cuda_fn = cuda.jit('void(int64[:], float64[:])')(atomic_add_double)
    cuda_fn[1, 32](idx, ary)
    wrap_fn = cuda.jit('void(int64[:], float64[:])')(atomic_add_double_wrap)
    wrap_fn[1, 32](idx, ary_wrap)
    gold = np.zeros(32, dtype=np.uint32)
    for i in range(idx.size):
        gold[idx[i]] += 1.0
    np.testing.assert_equal(ary, gold)
    np.testing.assert_equal(ary_wrap, gold)
    self.assertCorrectFloat64Atomics(cuda_fn)
    self.assertCorrectFloat64Atomics(wrap_fn)