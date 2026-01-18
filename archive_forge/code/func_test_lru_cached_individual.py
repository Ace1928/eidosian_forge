import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@pytest.mark.parametrize('method', bracket_methods)
def test_lru_cached_individual(self, method):
    a, b = (-1, 1)
    root, r = method(f_lrucached, a, b, full_output=True)
    assert r.converged
    assert_allclose(root, 0)