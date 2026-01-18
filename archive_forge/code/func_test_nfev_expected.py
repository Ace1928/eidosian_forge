import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('seed', (615655101, 3141866013, 238075752))
@pytest.mark.parametrize('use_min', (False, True))
@pytest.mark.parametrize('other_side', (False, True))
@pytest.mark.parametrize('fix_one_side', (False, True))
def test_nfev_expected(self, seed, use_min, other_side, fix_one_side):
    rng = np.random.default_rng(seed)
    a, d, factor = rng.random(size=3) * [100000.0, 10, 5]
    factor = 1 + factor
    b = a + d

    def f(x):
        f.count += 1
        return x
    if use_min:
        min = -rng.random()
        n = np.ceil(np.log(-(a - min) / min) / np.log(factor))
        l, u = (min + (a - min) * factor ** (-n), min + (a - min) * factor ** (-(n - 1)))
        kwargs = dict(a=a, b=b, factor=factor, min=min)
    else:
        n = np.ceil(np.log(b / d) / np.log(factor))
        l, u = (b - d * factor ** n, b - d * factor ** (n - 1))
        kwargs = dict(a=a, b=b, factor=factor)
    if other_side:
        kwargs['a'], kwargs['b'] = (-kwargs['b'], -kwargs['a'])
        l, u = (-u, -l)
        if 'min' in kwargs:
            kwargs['max'] = -kwargs.pop('min')
    if fix_one_side:
        if other_side:
            kwargs['min'] = -b
        else:
            kwargs['max'] = b
    f.count = 0
    res = zeros._bracket_root(f, **kwargs)
    if not fix_one_side:
        assert res.nfev == 2 * (res.nit + 1) == 2 * (f.count - 1) == 2 * (n + 1)
    else:
        assert res.nfev == res.nit + 1 + 1 == f.count - 1 + 1 == n + 1 + 1
    bracket = np.asarray([res.xl, res.xr])
    assert_allclose(bracket, (l, u))
    f_bracket = np.asarray([res.fl, res.fr])
    assert_allclose(f_bracket, f(bracket))
    assert res.xr > res.xl
    signs = np.sign(f_bracket)
    assert signs[0] == -signs[1]
    assert res.status == 0
    assert res.success