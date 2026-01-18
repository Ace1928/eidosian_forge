import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_newton_complex_gh10103():

    def f(z):
        return z - 1
    res = newton(f, 1 + 1j)
    assert_allclose(res, 1, atol=1e-12)
    res = root_scalar(f, x0=1 + 1j, x1=2 + 1.5j, method='secant')
    assert_allclose(res.root, 1, atol=1e-12)