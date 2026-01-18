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
@check_version(mpmath, '0.19')
def test_hyp0f1_gh_1609():
    vv = np.linspace(150, 180, 21)
    af = sc.hyp0f1(vv, 0.5)
    mf = np.array([mpmath.hyp0f1(v, 0.5) for v in vv])
    assert_allclose(af, mf.astype(float), rtol=1e-12)