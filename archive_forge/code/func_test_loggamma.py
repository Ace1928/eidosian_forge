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
def test_loggamma(self):

    def mpmath_loggamma(z):
        try:
            res = mpmath.loggamma(z)
        except ValueError:
            res = complex(np.nan, np.nan)
        return res
    assert_mpmath_equal(sc.loggamma, mpmath_loggamma, [ComplexArg()], nan_ok=False, distinguish_nan_and_inf=False, rtol=5e-14)