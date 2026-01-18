import math
import numpy as np
from numba import cuda
from numba.cuda.testing import unittest
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.cuda.random import \
def check_uniform(self, kernel_func, dtype):
    states = cuda.random.create_xoroshiro128p_states(32 * 2, seed=1)
    out = np.zeros(2 * 32 * 32, dtype=np.float32)
    kernel_func[2, 32](states, out, 32, UNIFORM)
    self.assertAlmostEqual(out.min(), 0.0, delta=0.001)
    self.assertAlmostEqual(out.max(), 1.0, delta=0.001)
    self.assertAlmostEqual(out.mean(), 0.5, delta=0.015)
    self.assertAlmostEqual(out.std(), 1.0 / (2 * math.sqrt(3)), delta=0.006)