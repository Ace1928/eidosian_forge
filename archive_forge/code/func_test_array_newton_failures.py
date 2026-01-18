import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_array_newton_failures():
    """Test that array newton fails as expected"""
    diameter = 0.1
    roughness = 0.00015
    rho = 988.1
    mu = 0.0005479
    u = 2.488
    reynolds_number = rho * u * diameter / mu

    def colebrook_eqn(darcy_friction, re, dia):
        return 1 / np.sqrt(darcy_friction) + 2 * np.log10(roughness / 3.7 / dia + 2.51 / re / np.sqrt(darcy_friction))
    with pytest.warns(RuntimeWarning):
        result = zeros.newton(colebrook_eqn, x0=[0.01, 0.2, 0.02223, 0.3], maxiter=2, args=[reynolds_number, diameter], full_output=True)
        assert not result.converged.all()
    with pytest.raises(RuntimeError):
        result = zeros.newton(colebrook_eqn, x0=[0.01] * 2, maxiter=2, args=[reynolds_number, diameter], full_output=True)