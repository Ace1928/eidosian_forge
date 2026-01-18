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
def test_input_validation(self):
    f = self.f1
    message = '`f` must be callable.'
    with pytest.raises(ValueError, match=message):
        _tanhsinh(42, 0, f.b)
    message = '...must be True or False.'
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, log=2)
    message = '...must be real numbers.'
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 1 + 1j, f.b)
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, atol='ekki')
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, rtol=pytest)
    message = '...must be non-negative and finite.'
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, rtol=-1)
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, atol=np.inf)
    message = '...may not be positive infinity.'
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, rtol=np.inf, log=True)
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, atol=np.inf, log=True)
    message = '...must be integers.'
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, maxlevel=object())
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, maxfun=1 + 1j)
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, minlevel='migratory coconut')
    message = '...must be non-negative.'
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, maxlevel=-1)
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, maxfun=-1)
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, minlevel=-1)
    message = '...must be callable.'
    with pytest.raises(ValueError, match=message):
        _tanhsinh(f, 0, f.b, callback='elderberry')