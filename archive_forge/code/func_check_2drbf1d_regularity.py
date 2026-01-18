import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def check_2drbf1d_regularity(function, atol):
    x = linspace(0, 10, 9)
    y0 = sin(x)
    y1 = cos(x)
    y = np.vstack([y0, y1]).T
    rbf = Rbf(x, y, function=function, mode='N-D')
    xi = linspace(0, 10, 100)
    yi = rbf(xi)
    msg = 'abs-diff: %f' % abs(yi - np.vstack([sin(xi), cos(xi)]).T).max()
    assert_(allclose(yi, np.vstack([sin(xi), cos(xi)]).T, atol=atol), msg)