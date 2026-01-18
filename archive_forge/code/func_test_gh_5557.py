import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_gh_5557():

    def f(x):
        if x < 0.5:
            return -0.1
        else:
            return x - 0.6
    atol = 0.51
    rtol = 4 * _FLOAT_EPS
    methods = [zeros.brentq, zeros.brenth]
    for method in methods:
        res = method(f, 0, 1, xtol=atol, rtol=rtol)
        assert_allclose(0.6, res, atol=atol, rtol=rtol)