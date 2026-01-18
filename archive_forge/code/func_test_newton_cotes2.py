import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
def test_newton_cotes2(self):
    """Test newton_cotes with points that are not evenly spaced."""
    x = np.array([0.0, 1.5, 2.0])
    y = x ** 2
    wts, errcoff = newton_cotes(x)
    exact_integral = 8.0 / 3
    numeric_integral = np.dot(wts, y)
    assert_almost_equal(numeric_integral, exact_integral)
    x = np.array([0.0, 1.4, 2.1, 3.0])
    y = x ** 2
    wts, errcoff = newton_cotes(x)
    exact_integral = 9.0
    numeric_integral = np.dot(wts, y)
    assert_almost_equal(numeric_integral, exact_integral)