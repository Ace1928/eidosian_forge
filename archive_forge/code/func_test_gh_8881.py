import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_gh_8881():
    """Test that Halley's method realizes that the 2nd order adjustment
    is too big and drops off to the 1st order adjustment."""
    n = 9

    def f(x):
        return power(x, 1.0 / n) - power(n, 1.0 / n)

    def fp(x):
        return power(x, (1.0 - n) / n) / n

    def fpp(x):
        return power(x, (1.0 - 2 * n) / n) * (1.0 / n) * (1.0 - n) / n
    x0 = 0.1
    rt, r = newton(f, x0, fprime=fp, full_output=True)
    assert r.converged
    rt, r = newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
    assert r.converged