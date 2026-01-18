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
def test_lgam1p(self):

    def param_filter(x):
        return np.where((np.floor(x) == x) & (x <= 0), False, True)

    def mp_lgam1p(z):
        return mpmath.loggamma(1 + z).real
    assert_mpmath_equal(_lgam1p, mp_lgam1p, [Arg()], rtol=1e-13, dps=100, param_filter=param_filter)