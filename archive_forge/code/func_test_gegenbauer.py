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
@pytest.mark.xfail(run=False)
def test_gegenbauer(self):
    assert_mpmath_equal(sc.eval_gegenbauer, exception_to_nan(mpmath.gegenbauer), [Arg(-1000.0, 1000.0), Arg(), Arg()])