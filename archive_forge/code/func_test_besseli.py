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
def test_besseli(self):
    assert_mpmath_equal(sc.iv, exception_to_nan(lambda v, z: mpmath.besseli(v, z, **HYPERKW)), [Arg(-1e+100, 1e+100), Arg()], atol=1e-270)