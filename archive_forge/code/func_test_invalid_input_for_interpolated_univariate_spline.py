import itertools
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_array_equal,
from pytest import raises as assert_raises
from numpy import array, diff, linspace, meshgrid, ones, pi, shape
from scipy.interpolate._fitpack_py import bisplrep, bisplev, splrep, spalde
from scipy.interpolate._fitpack2 import (UnivariateSpline,
def test_invalid_input_for_interpolated_univariate_spline(self):
    with assert_raises(ValueError) as info:
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5]
        InterpolatedUnivariateSpline(x_values, y_values)
    assert 'x and y should have a same length' in str(info.value)
    with assert_raises(ValueError) as info:
        x_values = [1, 2, 4, 6, 8.5]
        y_values = [0.5, 0.8, 1.3, 2.5, 2.8]
        w_values = [-1.0, 1.0, 1.0, 1.0]
        InterpolatedUnivariateSpline(x_values, y_values, w=w_values)
    assert 'x, y, and w should have a same length' in str(info.value)
    with assert_raises(ValueError) as info:
        bbox = -1
        InterpolatedUnivariateSpline(x_values, y_values, bbox=bbox)
    assert 'bbox shape should be (2,)' in str(info.value)
    with assert_raises(ValueError) as info:
        InterpolatedUnivariateSpline(x_values, y_values, k=6)
    assert 'k should be 1 <= k <= 5' in str(info.value)