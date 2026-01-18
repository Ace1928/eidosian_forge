import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def check_2drbf3d_interpolation(function):
    x = random.rand(50) * 4 - 2
    y = random.rand(50) * 4 - 2
    z = random.rand(50) * 4 - 2
    d0 = x * exp(-x ** 2 - y ** 2)
    d1 = y * exp(-y ** 2 - x ** 2)
    d = np.vstack([d0, d1]).T
    rbf = Rbf(x, y, z, d, epsilon=2, function=function, mode='N-D')
    di = rbf(x, y, z)
    di.shape = d.shape
    assert_array_almost_equal(di, d)