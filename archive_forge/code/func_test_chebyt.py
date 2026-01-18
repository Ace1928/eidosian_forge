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
@pytest.mark.xfail(run=False, reason='some cases in hyp2f1 not fully accurate')
def test_chebyt(self):
    assert_mpmath_equal(sc.eval_chebyt, lambda n, x: time_limited()(exception_to_nan(mpmath.chebyt))(n, x, **HYPERKW), [Arg(-101, 101), Arg()], n=10000)