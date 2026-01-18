import pytest
from numpy.testing import assert_allclose, assert_equal
import numpy as np
from math import pow
from scipy import optimize
def g5(self, x):
    dif = [0, 0]
    v1 = np.cos(x[0] + x[1])
    v2 = 2.0 * (x[0] - x[1])
    dif[0] = v1 + v2 - 1.5
    dif[1] = v1 - v2 + 2.5
    return dif