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
def test_ellipkinc_largephi(self):

    def ellipkinc(phi, m):
        return mpmath.ellippi(0, phi, m)
    assert_mpmath_equal(sc.ellipkinc, ellipkinc, [Arg(), Arg(b=1.0)], ignore_inf_sign=True)