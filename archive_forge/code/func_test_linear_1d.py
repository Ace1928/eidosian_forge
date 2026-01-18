import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_linear_1d(self):
    x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    y = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    z = [0, 0, 0, 2, 2, 2, 4, 4, 4]
    lut = SmoothBivariateSpline(x, y, z, kx=1, ky=1)
    assert_array_almost_equal(lut.get_knots(), ([1, 1, 3, 3], [1, 1, 3, 3]))
    assert_array_almost_equal(lut.get_coeffs(), [0, 0, 4, 4])
    assert_almost_equal(lut.get_residual(), 0.0)
    assert_array_almost_equal(lut([1, 1.5, 2], [1, 1.5]), [[0, 0], [1, 1], [2, 2]])