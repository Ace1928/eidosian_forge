import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_gh8904_zeroder_at_root_fails():
    """Test that Newton or Halley don't warn if zero derivative at root"""

    def f_zeroder_root(x):
        return x ** 3 - x ** 2
    r = zeros.newton(f_zeroder_root, x0=0)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0] * 10)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

    def fder(x):
        return 3 * x ** 2 - 2 * x

    def fder2(x):
        return 6 * x - 2
    r = zeros.newton(f_zeroder_root, x0=0, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=0, fprime=fder, fprime2=fder2)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0] * 10, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0] * 10, fprime=fder, fprime2=fder2)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=0.5, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
    r = zeros.newton(f_zeroder_root, x0=[0.5] * 10, fprime=fder)
    assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)