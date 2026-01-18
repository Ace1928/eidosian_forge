import warnings
from numpy import (logical_and, asarray, pi, zeros_like,
from numpy import (sqrt, exp, greater, less, cos, add, sin, less_equal,
from ._spline import cspline2d, sepfir2d
from ._signaltools import lfilter, sosfilt, lfiltic
from scipy.special import comb
from scipy._lib._util import float_factorial
from scipy.interpolate import BSpline
def piecefuncgen(num):
    Mk = order // 2 - num
    if Mk < 0:
        return 0
    coeffs = [(1 - 2 * (k % 2)) * float(comb(order + 1, k, exact=1)) / fval for k in range(Mk + 1)]
    shifts = [-bound - k for k in range(Mk + 1)]

    def thefunc(x):
        res = 0.0
        for k in range(Mk + 1):
            res += coeffs[k] * (x + shifts[k]) ** order
        return res
    return thefunc