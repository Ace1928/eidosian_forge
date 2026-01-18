import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_derivatives_grid(self):
    y = linspace(0.01, 2 * pi - 0.01, 7)
    x = linspace(0.01, pi - 0.01, 7)
    z = array([[1, 2, 1, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 3, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1], [1, 2, 2, 2, 1, 2, 1], [1, 2, 1, 2, 1, 2, 1]])
    lut = RectSphereBivariateSpline(x, y, z)
    y = linspace(0.02, 2 * pi - 0.02, 7)
    x = linspace(0.02, pi - 0.02, 7)
    assert_allclose(lut(x, y, dtheta=1), _numdiff_2d(lut, x, y, dx=1), rtol=0.0001, atol=0.0001)
    assert_allclose(lut(x, y, dphi=1), _numdiff_2d(lut, x, y, dy=1), rtol=0.0001, atol=0.0001)
    assert_allclose(lut(x, y, dtheta=1, dphi=1), _numdiff_2d(lut, x, y, dx=1, dy=1, eps=1e-06), rtol=0.001, atol=0.001)
    assert_array_equal(lut(x, y, dtheta=1), lut.partial_derivative(1, 0)(x, y))
    assert_array_equal(lut(x, y, dphi=1), lut.partial_derivative(0, 1)(x, y))
    assert_array_equal(lut(x, y, dtheta=1, dphi=1), lut.partial_derivative(1, 1)(x, y))
    assert_array_equal(lut(x, y, dtheta=1, grid=False), lut.partial_derivative(1, 0)(x, y, grid=False))
    assert_array_equal(lut(x, y, dphi=1, grid=False), lut.partial_derivative(0, 1)(x, y, grid=False))
    assert_array_equal(lut(x, y, dtheta=1, dphi=1, grid=False), lut.partial_derivative(1, 1)(x, y, grid=False))