import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_bilinearity(self):
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    z = [0, 7, 8, 3, 4, 7, 1, 3, 4]
    s = 0.1
    tx = [1 + s, 3 - s]
    ty = [1 + s, 3 - s]
    with suppress_warnings() as sup:
        sup.filter(UserWarning, '\nThe coefficients of the spline')
        lut = LSQBivariateSpline(x, y, z, tx, ty, kx=1, ky=1)
    tx, ty = lut.get_knots()
    for xa, xb in zip(tx[:-1], tx[1:]):
        for ya, yb in zip(ty[:-1], ty[1:]):
            for t in [0.1, 0.5, 0.9]:
                for s in [0.3, 0.4, 0.7]:
                    xp = xa * (1 - t) + xb * t
                    yp = ya * (1 - s) + yb * s
                    zp = +lut(xa, ya) * (1 - t) * (1 - s) + lut(xb, ya) * t * (1 - s) + lut(xa, yb) * (1 - t) * s + lut(xb, yb) * t * s
                    assert_almost_equal(lut(xp, yp), zp)