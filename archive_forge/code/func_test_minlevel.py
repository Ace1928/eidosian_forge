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
@pytest.mark.parametrize('maxlevel', range(4))
def test_minlevel(self, maxlevel):

    def f(x):
        f.calls += 1
        f.feval += np.size(x)
        f.x = np.concatenate((f.x, x.ravel()))
        return self.f2(x)
    f.feval, f.calls, f.x = (0, 0, np.array([]))
    ref = _tanhsinh(f, 0, self.f2.b, minlevel=0, maxlevel=maxlevel)
    ref_x = np.sort(f.x)
    for minlevel in range(0, maxlevel + 1):
        f.feval, f.calls, f.x = (0, 0, np.array([]))
        options = dict(minlevel=minlevel, maxlevel=maxlevel)
        res = _tanhsinh(f, 0, self.f2.b, **options)
        assert_allclose(res.integral, ref.integral, rtol=4e-16)
        assert_allclose(res.error, ref.error, atol=4e-16 * ref.integral)
        assert res.nfev == f.feval == len(f.x)
        assert f.calls == maxlevel - minlevel + 1 + 1
        assert res.status == ref.status
        assert_equal(ref_x, np.sort(f.x))