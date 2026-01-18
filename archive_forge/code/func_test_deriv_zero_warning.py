import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_deriv_zero_warning(self):

    def func(x):
        return x ** 2 - 2.0

    def dfunc(x):
        return 2 * x
    assert_warns(RuntimeWarning, zeros.newton, func, 0.0, dfunc, disp=False)
    with pytest.raises(RuntimeError, match='Derivative was zero'):
        zeros.newton(func, 0.0, dfunc)