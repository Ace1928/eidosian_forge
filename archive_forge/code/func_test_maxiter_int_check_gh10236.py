import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('method', all_methods)
def test_maxiter_int_check_gh10236(method):
    message = "'float' object cannot be interpreted as an integer"
    with pytest.raises(TypeError, match=message):
        method(f1, 0.0, 1.0, maxiter=72.45)