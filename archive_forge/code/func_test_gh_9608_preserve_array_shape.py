import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_gh_9608_preserve_array_shape():
    """
    Test that shape is preserved for array inputs even if fprime or fprime2 is
    scalar
    """

    def f(x):
        return x ** 2

    def fp(x):
        return 2 * x

    def fpp(x):
        return 2
    x0 = np.array([-2], dtype=np.float32)
    rt, r = newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
    assert r.converged
    x0_array = np.array([-2, -3], dtype=np.float32)
    with pytest.raises(IndexError):
        result = zeros.newton(f, x0_array, fprime=fp, fprime2=fpp, full_output=True)

    def fpp_array(x):
        return np.full(np.shape(x), 2, dtype=np.float32)
    result = zeros.newton(f, x0_array, fprime=fp, fprime2=fpp_array, full_output=True)
    assert result.converged.all()