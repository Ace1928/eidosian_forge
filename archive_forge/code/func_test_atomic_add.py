import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_add(self):
    ary = np.random.randint(0, 32, size=32).astype(np.uint32)
    ary_wrap = ary.copy()
    orig = ary.copy()
    cuda_atomic_add = cuda.jit('void(uint32[:])')(atomic_add)
    cuda_atomic_add[1, 32](ary)
    cuda_atomic_add_wrap = cuda.jit('void(uint32[:])')(atomic_add_wrap)
    cuda_atomic_add_wrap[1, 32](ary_wrap)
    gold = np.zeros(32, dtype=np.uint32)
    for i in range(orig.size):
        gold[orig[i]] += 1
    self.assertTrue(np.all(ary == gold))
    self.assertTrue(np.all(ary_wrap == gold))