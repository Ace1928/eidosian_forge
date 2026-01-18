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
def test_masked(self):
    x = np.arange(5)
    y = x * x
    mask = x == 2
    ym = np.ma.array(y, mask=mask)
    r = 13.0
    assert_allclose(trapezoid(ym, x), r)
    xm = np.ma.array(x, mask=mask)
    assert_allclose(trapezoid(ym, xm), r)
    xm = np.ma.array(x, mask=mask)
    assert_allclose(trapezoid(y, xm), r)