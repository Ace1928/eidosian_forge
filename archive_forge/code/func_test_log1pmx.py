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
def test_log1pmx(self):
    assert_mpmath_equal(_log1pmx, lambda x: mpmath.log(x + 1) - x, [Arg()], dps=60, rtol=1e-14)