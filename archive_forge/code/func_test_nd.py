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
@pytest.mark.parametrize('axis', np.arange(-3, 3))
@pytest.mark.parametrize('x_ndim', (1, 3))
@pytest.mark.parametrize('x_len', (1, 2, 7))
@pytest.mark.parametrize('i_ndim', (None, 0, 3))
@pytest.mark.parametrize('dx', (None, True))
def test_nd(self, axis, x_ndim, x_len, i_ndim, dx):
    rng = np.random.default_rng(82456839535679456794)
    shape = [5, 6, x_len]
    shape[axis], shape[-1] = (shape[-1], shape[axis])
    shape_len_1 = shape.copy()
    shape_len_1[axis] = 1
    i_shape = shape_len_1 if i_ndim == 3 else ()
    y = rng.random(size=shape)
    x, dx = (None, None)
    if dx:
        dx = rng.random(size=shape_len_1) if x_ndim > 1 else rng.random()
    else:
        x = np.sort(rng.random(size=shape), axis=axis) if x_ndim > 1 else np.sort(rng.random(size=shape[axis]))
    initial = None if i_ndim is None else rng.random(size=i_shape)
    res = cumulative_simpson(y, x=x, dx=dx, initial=initial, axis=axis)
    ref = cumulative_simpson_nd_reference(y, x=x, dx=dx, initial=initial, axis=axis)
    np.testing.assert_allclose(res, ref, rtol=1e-15)