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
def test_besselj(self):
    assert_mpmath_equal(sc.jv, exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg(-1000.0, 1000.0)], ignore_inf_sign=True)
    assert_mpmath_equal(sc.jv, exception_to_nan(lambda v, z: mpmath.besselj(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg(-100000000.0, 100000000.0)], ignore_inf_sign=True, rtol=1e-05)