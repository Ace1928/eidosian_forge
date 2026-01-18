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
def test_lanczos_sum_expg_scaled(self):
    maxgamma = 171.6243769563027
    e = np.exp(1)
    g = 6.02468004077673

    def gamma(x):
        with np.errstate(over='ignore'):
            fac = ((x + g - 0.5) / e) ** (x - 0.5)
            if fac != np.inf:
                res = fac * _lanczos_sum_expg_scaled(x)
            else:
                fac = ((x + g - 0.5) / e) ** (0.5 * (x - 0.5))
                res = fac * _lanczos_sum_expg_scaled(x)
                res *= fac
        return res
    assert_mpmath_equal(gamma, mpmath.gamma, [Arg(0, maxgamma, inclusive_a=False)], rtol=1e-13)