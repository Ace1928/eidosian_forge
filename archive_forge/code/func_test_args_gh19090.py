import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('kwargs', [dict(), {'method': 'newton'}])
def test_args_gh19090(self, kwargs):

    def f(x, a, b):
        assert a == 3
        assert b == 1
        return x ** a - b
    res = optimize.root_scalar(f, x0=3, args=(3, 1), **kwargs)
    assert res.converged
    assert_allclose(res.root, 1)