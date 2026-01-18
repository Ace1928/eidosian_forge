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
def test_boxcox1p(self):

    def mp_boxcox1p(x, lmbda):
        x = mpmath.mp.mpf(x)
        lmbda = mpmath.mp.mpf(lmbda)
        one = mpmath.mp.mpf(1)
        if lmbda == 0:
            return mpmath.mp.log(one + x)
        else:
            return mpmath.mp.powm1(one + x, lmbda) / lmbda
    assert_mpmath_equal(sc.boxcox1p, exception_to_nan(mp_boxcox1p), [Arg(a=-1, inclusive_a=False), Arg()], n=200, dps=60, rtol=1e-13)