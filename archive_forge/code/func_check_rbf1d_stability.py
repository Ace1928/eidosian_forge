import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def check_rbf1d_stability(function):
    np.random.seed(1234)
    x = np.linspace(0, 10, 50)
    z = x + 4.0 * np.random.randn(len(x))
    rbf = Rbf(x, z, function=function)
    xi = np.linspace(0, 10, 1000)
    yi = rbf(xi)
    assert_(np.abs(yi - xi).max() / np.abs(z - x).max() < 1.1)