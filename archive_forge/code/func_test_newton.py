import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_newton(self):
    for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
        x = zeros.newton(f, 3, tol=1e-06)
        assert_allclose(f(x), 0, atol=1e-06)
        x = zeros.newton(f, 3, x1=5, tol=1e-06)
        assert_allclose(f(x), 0, atol=1e-06)
        x = zeros.newton(f, 3, fprime=f_1, tol=1e-06)
        assert_allclose(f(x), 0, atol=1e-06)
        x = zeros.newton(f, 3, fprime=f_1, fprime2=f_2, tol=1e-06)
        assert_allclose(f(x), 0, atol=1e-06)