import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def check_2drbf2d_interpolation(function):
    x = random.rand(50) * 4 - 2
    y = random.rand(50) * 4 - 2
    z0 = x * exp(-x ** 2 - 1j * y ** 2)
    z1 = y * exp(-y ** 2 - 1j * x ** 2)
    z = np.vstack([z0, z1]).T
    rbf = Rbf(x, y, z, epsilon=2, function=function, mode='N-D')
    zi = rbf(x, y)
    zi.shape = z.shape
    assert_array_almost_equal(z, zi)