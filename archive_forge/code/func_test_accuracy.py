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
@pytest.mark.parametrize('ref', (0.5, [0.4, 0.6]))
@pytest.mark.parametrize('case', stats._distr_params.distcont)
def test_accuracy(self, ref, case):
    distname, params = case
    if distname in {'dgamma', 'dweibull', 'laplace', 'kstwo'}:
        pytest.skip('tanh-sinh is not great for non-smooth integrands')
    dist = getattr(stats, distname)(*params)
    x = dist.interval(ref)
    res = _tanhsinh(dist.pdf, *x)
    assert_allclose(res.integral, ref)