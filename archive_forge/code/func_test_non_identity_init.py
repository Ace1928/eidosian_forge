import numpy as np
from numba import cuda
from numba.core.config import ENABLE_CUDASIM
from numba.cuda.testing import CUDATestCase
import unittest
def test_non_identity_init(self):
    init = 3
    A = np.arange(10, dtype=np.float64) + 1
    expect = A.sum() + init
    got = sum_reduce(A, init=init)
    self.assertEqual(expect, got)