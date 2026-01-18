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
def test_chi_complex(self):

    def chi(z):
        return sc.shichi(z)[1]
    assert_mpmath_equal(chi, mpmath.chi, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-12)