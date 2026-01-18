import pytest
from functools import lru_cache
from numpy.testing import (assert_warns, assert_,
import numpy as np
from numpy import finfo, power, nan, isclose, sqrt, exp, sin, cos
from scipy import stats, optimize
from scipy.optimize import (_zeros_py as zeros, newton, root_scalar,
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from scipy.optimize._tstutils import get_tests, functions as tstutils_functions
def test_halley_collections(self):
    known_fail = ['aps.12.06', 'aps.12.07', 'aps.12.08', 'aps.12.09', 'aps.12.10', 'aps.12.11', 'aps.12.12', 'aps.12.13', 'aps.12.14', 'aps.12.15', 'aps.12.16', 'aps.12.17', 'aps.12.18', 'aps.13.00']
    for collection in ['aps', 'complex']:
        self.run_collection(collection, zeros.newton, 'halley', smoothness=2, known_fail=known_fail)