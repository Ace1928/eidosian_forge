import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_array_newton_zero_der_failures(self):
    assert_warns(RuntimeWarning, zeros.newton, lambda y: y ** 2 - 2, [0.0, 0.0], lambda y: 2 * y)
    with pytest.warns(RuntimeWarning):
        results = zeros.newton(lambda y: y ** 2 - 2, [0.0, 0.0], lambda y: 2 * y, full_output=True)
        assert_allclose(results.root, 0)
        assert results.zero_der.all()
        assert not results.converged.any()