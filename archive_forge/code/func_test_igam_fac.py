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
@pytest.mark.xfail_on_32bit('mpmath issue gh-342: unsupported operand mpz, long for pow')
def test_igam_fac(self):

    def mp_igam_fac(a, x):
        return mpmath.power(x, a) * mpmath.exp(-x) / mpmath.gamma(a)
    assert_mpmath_equal(_igam_fac, mp_igam_fac, [Arg(0, 100000000000000.0, inclusive_a=False), Arg(0, 100000000000000.0)], rtol=1e-10)