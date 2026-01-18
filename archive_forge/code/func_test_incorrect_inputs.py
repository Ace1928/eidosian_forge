import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_incorrect_inputs(self):
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 3, 4])
    xc = np.array([1 + 1j, 2, 3, 4])
    xn = np.array([np.nan, 2, 3, 4])
    xo = np.array([2, 1, 3, 4])
    yn = np.array([np.nan, 2, 3, 4])
    y3 = [1, 2, 3]
    x1 = [1]
    y1 = [1]
    assert_raises(ValueError, CubicSpline, xc, y)
    assert_raises(ValueError, CubicSpline, xn, y)
    assert_raises(ValueError, CubicSpline, x, yn)
    assert_raises(ValueError, CubicSpline, xo, y)
    assert_raises(ValueError, CubicSpline, x, y3)
    assert_raises(ValueError, CubicSpline, x[:, np.newaxis], y)
    assert_raises(ValueError, CubicSpline, x1, y1)
    wrong_bc = [('periodic', 'clamped'), ((2, 0), (3, 10)), ((1, 0),), (0.0, 0.0), 'not-a-typo']
    for bc_type in wrong_bc:
        assert_raises(ValueError, CubicSpline, x, y, 0, bc_type, True)
    Y = np.c_[y, y]
    bc1 = ('clamped', (1, 0))
    bc2 = ('clamped', (1, [0, 0, 0]))
    bc3 = ('clamped', (1, [[0, 0]]))
    assert_raises(ValueError, CubicSpline, x, Y, 0, bc1, True)
    assert_raises(ValueError, CubicSpline, x, Y, 0, bc2, True)
    assert_raises(ValueError, CubicSpline, x, Y, 0, bc3, True)
    assert_raises(ValueError, CubicSpline, x, y, 0, 'periodic', True)