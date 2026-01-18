import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def check_rbf2d_interpolation(function):
    x = random.rand(50, 1) * 4 - 2
    y = random.rand(50, 1) * 4 - 2
    z = x * exp(-x ** 2 - 1j * y ** 2)
    rbf = Rbf(x, y, z, epsilon=2, function=function)
    zi = rbf(x, y)
    zi.shape = x.shape
    assert_array_almost_equal(z, zi)