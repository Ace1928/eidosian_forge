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
def test_dn_quarter_period():

    def dn(u, m):
        return sc.ellipj(u, m)[2]

    def mpmath_dn(u, m):
        return float(mpmath.ellipfun('dn', u=u, m=m))
    m = np.linspace(0, 1, 20)
    du = np.r_[-np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10)]
    dataset = []
    for m0 in m:
        u0 = float(mpmath.ellipk(m0))
        for du0 in du:
            p = u0 + du0
            dataset.append((p, m0, mpmath_dn(p, m0)))
    dataset = np.asarray(dataset)
    FuncData(dn, dataset, (0, 1), 2, rtol=1e-10).check()