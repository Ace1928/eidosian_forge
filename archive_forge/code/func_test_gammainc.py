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
def test_gammainc(self):
    assert_mpmath_equal(sc.gammainc, lambda z, b: mpmath.gammainc(z, b=b, regularized=True), [Arg(0, 10000.0, inclusive_a=False), Arg(0, 10000.0)], nan_ok=False, rtol=1e-11)