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
def test_eulernum(self):
    assert_mpmath_equal(lambda n: sc.euler(n)[-1], mpmath.eulernum, [IntArg(1, 10000)], n=10000)