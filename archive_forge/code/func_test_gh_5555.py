import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_gh_5555():
    root = 0.1

    def f(x):
        return x - root
    methods = [zeros.bisect, zeros.ridder]
    xtol = rtol = TOL
    for method in methods:
        res = method(f, -100000000.0, 10000000.0, xtol=xtol, rtol=rtol)
        assert_allclose(root, res, atol=xtol, rtol=rtol, err_msg='method %s' % method.__name__)