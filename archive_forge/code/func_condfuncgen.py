import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def condfuncgen(num, val1, val2):
    if num == 0:
        return lambda x: logical_and(less_equal(x, val1), greater_equal(x, val2))
    elif num == 2:
        return lambda x: less_equal(x, val2)
    else:
        return lambda x: logical_and(less(x, val1), greater_equal(x, val2))