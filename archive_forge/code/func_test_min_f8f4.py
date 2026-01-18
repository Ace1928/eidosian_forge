import numpy as np
from numba import cuda, float64
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
def test_min_f8f4(self):
    self._run(builtin_min, np.minimum, 'min.f64', np.float64, np.float32)