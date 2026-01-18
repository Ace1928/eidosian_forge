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
def test_log_ndtr(self):
    assert_mpmath_equal(sc.log_ndtr, exception_to_nan(lambda z: mpmath.log(mpmath.ncdf(z))), [Arg()], n=600, dps=300, rtol=1e-13)