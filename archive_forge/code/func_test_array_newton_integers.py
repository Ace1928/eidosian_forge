import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_array_newton_integers(self):
    x = zeros.newton(lambda y, z: z - y ** 2, [4.0] * 2, args=([15.0, 17.0],))
    assert_allclose(x, (3.872983346207417, 4.123105625617661))
    x = zeros.newton(lambda y, z: z - y ** 2, [4] * 2, args=([15, 17],))
    assert_allclose(x, (3.872983346207417, 4.123105625617661))