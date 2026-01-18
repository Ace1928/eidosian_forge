import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_array_newton_complex(self):

    def f(x):
        return x + 1 + 1j

    def fprime(x):
        return 1.0
    t = np.full(4, 1j)
    x = zeros.newton(f, t, fprime=fprime)
    assert_allclose(f(x), 0.0)
    t = np.ones(4)
    x = zeros.newton(f, t, fprime=fprime)
    assert_allclose(f(x), 0.0)
    x = zeros.newton(f, t)
    assert_allclose(f(x), 0.0)