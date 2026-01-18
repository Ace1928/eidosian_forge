import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def test_atomic_add_float(self):
    ary = np.random.randint(0, 32, size=32).astype(np.float32)
    ary_wrap = ary.copy()
    orig = ary.copy().astype(np.intp)
    cuda_atomic_add_float = cuda.jit('void(float32[:])')(atomic_add_float)
    cuda_atomic_add_float[1, 32](ary)
    add_float_wrap = cuda.jit('void(float32[:])')(atomic_add_float_wrap)
    add_float_wrap[1, 32](ary_wrap)
    gold = np.zeros(32, dtype=np.uint32)
    for i in range(orig.size):
        gold[orig[i]] += 1.0
    self.assertTrue(np.all(ary == gold))
    self.assertTrue(np.all(ary_wrap == gold))