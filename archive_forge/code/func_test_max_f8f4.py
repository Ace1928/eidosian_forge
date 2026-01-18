import numpy as np
from numba import cuda, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_max_f8f4(self):
    self._run(builtin_max, np.maximum, 'max.f64', np.float64, np.float32)