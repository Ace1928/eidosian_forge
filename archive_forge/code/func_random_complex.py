import math
import numpy as np
from numba import cuda, float64, int8, int32, void
from numba.cuda.testing import unittest, CUDATestCase
def random_complex(N):
    np.random.seed(123)
    return np.random.random(1) + np.random.random(1) * 1j