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
def test_ndtr_complex(self):
    assert_mpmath_equal(sc.ndtr, lambda z: mpmath.erfc(-z / np.sqrt(2.0)) / 2.0, [ComplexArg(a=complex(-10000, -10000), b=complex(10000, 10000))], n=400)