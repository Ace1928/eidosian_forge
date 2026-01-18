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
def test_ellipfun_dn(self):
    assert_mpmath_equal(lambda u, m: sc.ellipj(u, m)[2], lambda u, m: mpmath.ellipfun('dn', u=u, m=m), [Arg(-1000000.0, 1000000.0), Arg(a=0, b=1)], rtol=1e-08)