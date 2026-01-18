import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
@np.vectorize
def bracket_root_single(a, b, min, max, factor, p):
    return zeros._bracket_root(self.f, a, b, min=min, max=max, factor=factor, args=(p,), maxiter=maxiter)