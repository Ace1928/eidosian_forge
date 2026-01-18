import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_evaluate(self):
    y = linspace(0.01, 2 * pi - 0.01, 7)
    x = linspace(0.01, pi - 0.01, 7)
    z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 3, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1]])
    lut = RectSphereBivariateSpline(x, y, z)
    yi = [0.2, 1, 2.3, 2.35, 3.0, 3.99, 5.25]
    xi = [1.5, 0.4, 1.1, 0.45, 0.2345, 1.0, 0.0001]
    zi = lut.ev(xi, yi)
    zi2 = array([lut(xp, yp)[0, 0] for xp, yp in zip(xi, yi)])
    assert_almost_equal(zi, zi2)