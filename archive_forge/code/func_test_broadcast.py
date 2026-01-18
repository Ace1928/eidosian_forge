import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_broadcast(self):
    x = array([1, 2, 3, 4, 5])
    y = array([1, 2, 3, 4, 5])
    z = array([[1, 2, 1, 2, 1], [1, 2, 1, 2, 1], [1, 2, 3, 2, 1], [1, 2, 2, 2, 1], [1, 2, 1, 2, 1]])
    lut = RectBivariateSpline(x, y, z)
    assert_allclose(lut(x, y), lut(x[:, None], y[None, :], grid=False))