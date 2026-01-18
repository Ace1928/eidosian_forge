import warnings
import io
import numpy as np
from numpy.testing import (
from pytest import raises as assert_raises
import pytest
from scipy.interpolate import (
def test_CubicHermiteSpline_correctness():
    x = [0, 2, 7]
    y = [-1, 2, 3]
    dydx = [0, 3, 7]
    s = CubicHermiteSpline(x, y, dydx)
    assert_allclose(s(x), y, rtol=1e-15)
    assert_allclose(s(x, 1), dydx, rtol=1e-15)