import sys
import warnings
from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises
import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function
def test_geterr():
    err = sc.geterr()
    for key, value in err.items():
        assert_(key in _sf_error_code_map)
        assert_(value in _sf_error_actions)