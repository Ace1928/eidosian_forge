import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_CubicHermiteSpline_error_handling():
    x = [1, 2, 3]
    y = [0, 3, 5]
    dydx = [1, -1, 2, 3]
    assert_raises(ValueError, CubicHermiteSpline, x, y, dydx)
    dydx_with_nan = [1, 0, np.nan]
    assert_raises(ValueError, CubicHermiteSpline, x, y, dydx_with_nan)