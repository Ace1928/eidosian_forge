import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_gh13407():

    def f(x):
        return x ** 3 - 2 * x - 5
    xtol = 1e-300
    eps = np.finfo(float).eps
    x1 = zeros.toms748(f, 1e-10, 10000000000.0, xtol=xtol, rtol=1 * eps)
    f1 = f(x1)
    x4 = zeros.toms748(f, 1e-10, 10000000000.0, xtol=xtol, rtol=4 * eps)
    f4 = f(x4)
    assert f1 < f4
    message = f'rtol too small \\({eps / 2:g} < {eps:g}\\)'
    with pytest.raises(ValueError, match=message):
        zeros.toms748(f, 1e-10, 10000000000.0, xtol=xtol, rtol=eps / 2)