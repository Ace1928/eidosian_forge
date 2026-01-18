import numpy as np
from numpy.testing import assert_, assert_allclose
from numpy import pi
import pytest
import itertools
from scipy._lib import _pep440
import scipy.special as sc
from scipy.special._testutils import (
from scipy.special._mptestutils import (
from scipy.special._ufuncs import (
def test_bessely_int(self):

    def mpbessely(v, x):
        r = float(mpmath.bessely(v, x))
        if abs(r) == 0 and x == 0:
            return np.nan
        return r
    assert_mpmath_equal(lambda v, z: sc.yn(int(v), z), exception_to_nan(mpbessely), [IntArg(-1000, 1000), Arg(-100000000.0, 100000000.0)])