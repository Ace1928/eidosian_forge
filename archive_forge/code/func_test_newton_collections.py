import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_newton_collections(self):
    known_fail = ['aps.13.00']
    known_fail += ['aps.12.05', 'aps.12.17']
    for collection in ['aps', 'complex']:
        self.run_collection(collection, zeros.newton, 'newton', smoothness=2, known_fail=known_fail)