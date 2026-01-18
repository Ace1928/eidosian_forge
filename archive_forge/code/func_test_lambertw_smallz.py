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
@pytest.mark.slow
@check_version(mpmath, '0.19')
def test_lambertw_smallz():
    x, y = (np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))
    x, y = np.meshgrid(x, y)
    z = (x + 1j * y).flatten()
    dataset = np.asarray([(z0, complex(mpmath.lambertw(z0))) for z0 in z])
    FuncData(sc.lambertw, dataset, 0, 1, rtol=1e-13).check()