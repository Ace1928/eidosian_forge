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
def test_log_ndtr_complex(self):
    assert_mpmath_equal(sc.log_ndtr, exception_to_nan(lambda z: mpmath.log(mpmath.erfc(-z / np.sqrt(2.0)) / 2.0)), [ComplexArg(a=complex(-10000, -100), b=complex(10000, 100))], n=200, dps=300)