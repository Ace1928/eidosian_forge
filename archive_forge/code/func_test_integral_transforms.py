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
@pytest.mark.parametrize('limits, ref', [[(0, np.inf), 0.5], [(-np.inf, 0), 0.5], [(-np.inf, np.inf), 1], [(np.inf, -np.inf), -1], [(1, -1), stats.norm.cdf(-1) - stats.norm.cdf(1)]])
def test_integral_transforms(self, limits, ref):
    dist = stats.norm()
    res = _tanhsinh(dist.pdf, *limits)
    assert_allclose(res.integral, ref)
    logres = _tanhsinh(dist.logpdf, *limits, log=True)
    assert_allclose(np.exp(logres.integral), ref)
    assert np.issubdtype(logres.integral.dtype, np.floating) if ref > 0 else np.issubdtype(logres.integral.dtype, np.complexfloating)
    assert_allclose(np.exp(logres.error), res.error, atol=1e-16)