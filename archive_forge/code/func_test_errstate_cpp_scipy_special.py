import sys
import warnings
from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises
import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function
def test_errstate_cpp_scipy_special():
    olderr = sc.geterr()
    with sc.errstate(singular='raise'):
        with assert_raises(sc.SpecialFunctionError):
            sc.lambertw(0, 1)
    assert_equal(olderr, sc.geterr())