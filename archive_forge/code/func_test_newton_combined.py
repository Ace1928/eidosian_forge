import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_newton_combined(self):

    def f1(x):
        return x ** 2 - 2 * x - 1

    def f1_1(x):
        return 2 * x - 2

    def f1_2(x):
        return 2.0 + 0 * x

    def f1_and_p_and_pp(x):
        return (x ** 2 - 2 * x - 1, 2 * x - 2, 2.0)
    sol0 = root_scalar(f1, method='newton', x0=3, fprime=f1_1)
    sol = root_scalar(f1_and_p_and_pp, method='newton', x0=3, fprime=True)
    assert_allclose(sol0.root, sol.root, atol=1e-08)
    assert_equal(2 * sol.function_calls, sol0.function_calls)
    sol0 = root_scalar(f1, method='halley', x0=3, fprime=f1_1, fprime2=f1_2)
    sol = root_scalar(f1_and_p_and_pp, method='halley', x0=3, fprime2=True)
    assert_allclose(sol0.root, sol.root, atol=1e-08)
    assert_equal(3 * sol.function_calls, sol0.function_calls)