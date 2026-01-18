import numpy as np
from numpy.testing import (assert_, assert_array_almost_equal,
from numpy import linspace, sin, cos, random, exp, allclose
from scipy.interpolate._rbf import Rbf
def check_rbf1d_regularity(function, atol):
    x = linspace(0, 10, 9)
    y = sin(x)
    rbf = Rbf(x, y, function=function)
    xi = linspace(0, 10, 100)
    yi = rbf(xi)
    msg = 'abs-diff: %f' % abs(yi - sin(xi)).max()
    assert_(allclose(yi, sin(xi), atol=atol), msg)