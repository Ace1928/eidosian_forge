import numpy as np
import math
from numba import cuda, double, void
from numba.cuda.testing import unittest, CUDATestCase
def cnd(d):
    K = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    ret_val = RSQRT2PI * np.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    return np.where(d > 0, 1.0 - ret_val, ret_val)