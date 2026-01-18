import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_array_almost_equal
from scipy.special import lambertw
from numpy import nan, inf, pi, e, isnan, log, r_, array, complex128
from scipy.special._testutils import FuncData
@pytest.mark.parametrize('z', [1e-316, -2e-320j, -5e-318 + 1e-320j])
def test_lambertw_subnormal_k0(z):
    w = lambertw(z)
    assert w == z