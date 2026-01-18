import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('maximum_iterations,flag_expected', [(10, zeros.CONVERR), (100, zeros.CONVERGED)])
def test_gh9254_flag_if_maxiter_exceeded(maximum_iterations, flag_expected):
    """
    Test that if the maximum iterations is exceeded that the flag is not
    converged.
    """
    result = zeros.brentq(lambda x: ((1.2 * x - 2.3) * x + 3.4) * x - 4.5, -30, 30, (), 1e-06, 1e-06, maximum_iterations, full_output=True, disp=False)
    assert result[1].flag == flag_expected
    if flag_expected == zeros.CONVERR:
        assert result[1].iterations == maximum_iterations
    elif flag_expected == zeros.CONVERGED:
        assert result[1].iterations < maximum_iterations