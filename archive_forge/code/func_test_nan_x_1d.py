import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@pytest.mark.parametrize('method', ['nearest', 'linear'])
def test_nan_x_1d(self, method):
    f = RegularGridInterpolator(([1, 2, 3],), [10, 20, 30], fill_value=1, bounds_error=False, method=method)
    assert np.isnan(f([np.nan]))
    rng = np.random.default_rng(8143215468)
    x = rng.random(size=100) * 4
    i = rng.random(size=100) > 0.5
    x[i] = np.nan
    with np.errstate(invalid='ignore'):
        res = f(x)
    assert_equal(res[i], np.nan)
    assert_equal(res[~i], f(x[~i]))
    x = [1, 2, 3]
    y = [1]
    data = np.ones((3, 1))
    f = RegularGridInterpolator((x, y), data, fill_value=1, bounds_error=False, method=method)
    assert np.isnan(f([np.nan, 1]))
    assert np.isnan(f([1, np.nan]))