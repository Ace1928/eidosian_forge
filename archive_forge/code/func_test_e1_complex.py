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
def test_e1_complex(self):
    assert_mpmath_equal(sc.exp1, mpmath.e1, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-11)
    assert_mpmath_equal(sc.exp1, mpmath.e1, (np.linspace(-50, 50, 171)[:, None] + np.r_[0, np.logspace(-3, 2, 61), -np.logspace(-3, 2, 11)] * 1j).ravel(), rtol=1e-11)
    assert_mpmath_equal(sc.exp1, mpmath.e1, np.linspace(-50, -35, 10000) + 0j, rtol=1e-11)