import sys
import warnings
from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises
import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function
def test_errstate_all_but_one():
    olderr = sc.geterr()
    with sc.errstate(all='raise', singular='ignore'):
        sc.gammaln(0)
        with assert_raises(sc.SpecialFunctionError):
            sc.spence(-1.0)
    assert_equal(olderr, sc.geterr())