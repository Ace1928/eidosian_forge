import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def check_2drbf1d_interpolation(function):
    x = linspace(0, 10, 9)
    y0 = sin(x)
    y1 = cos(x)
    y = np.vstack([y0, y1]).T
    rbf = Rbf(x, y, function=function, mode='N-D')
    yi = rbf(x)
    assert_array_almost_equal(y, yi)
    assert_almost_equal(rbf(float(x[0])), y[0])