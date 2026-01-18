import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('method', bracket_methods)
@pytest.mark.parametrize('function', tstutils_functions)
def test_basic_root_scalar(self, method, function):
    a, b = (0.5, sqrt(3))
    r = root_scalar(function, method=method.__name__, bracket=[a, b], x0=a, xtol=self.xtol, rtol=self.rtol)
    assert r.converged
    assert_allclose(r.root, 1.0, atol=self.xtol, rtol=self.rtol)
    assert r.method == method.__name__