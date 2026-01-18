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
@pytest.mark.slow
def test_digamma_roots():
    root = mpmath.findroot(mpmath.digamma, 1.5)
    roots = [float(root)]
    root = mpmath.findroot(mpmath.digamma, -0.5)
    roots.append(float(root))
    roots = np.array(roots)
    dx = np.r_[-0.24, -np.logspace(-1, -15, 10), 0, np.logspace(-15, -1, 10), 0.24]
    dy = dx.copy()
    dx, dy = np.meshgrid(dx, dy)
    dz = dx + 1j * dy
    z = (roots + np.dstack((dz,) * roots.size)).flatten()
    with mpmath.workdps(30):
        dataset = [(z0, complex(mpmath.digamma(z0))) for z0 in z]
    dataset = np.array(dataset)
    FuncData(sc.digamma, dataset, 0, 1, rtol=1e-14).check()