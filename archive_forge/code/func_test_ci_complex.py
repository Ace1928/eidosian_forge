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
def test_ci_complex(self):

    def ci(z):
        return sc.sici(z)[1]
    assert_mpmath_equal(ci, mpmath.ci, [ComplexArg(complex(-100000000.0, -np.inf), complex(100000000.0, np.inf))], rtol=1e-08)