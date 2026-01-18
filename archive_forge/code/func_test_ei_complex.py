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
def test_ei_complex(self):
    assert_mpmath_equal(sc.expi, mpmath.ei, [ComplexArg(complex(-np.inf, -100000000.0), complex(np.inf, 100000000.0))], rtol=1e-09)