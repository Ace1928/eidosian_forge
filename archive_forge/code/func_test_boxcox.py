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
def test_boxcox(self):

    def mp_boxcox(x, lmbda):
        x = mpmath.mp.mpf(x)
        lmbda = mpmath.mp.mpf(lmbda)
        if lmbda == 0:
            return mpmath.mp.log(x)
        else:
            return mpmath.mp.powm1(x, lmbda) / lmbda
    assert_mpmath_equal(sc.boxcox, exception_to_nan(mp_boxcox), [Arg(a=0, inclusive_a=False), Arg()], n=200, dps=60, rtol=1e-13)