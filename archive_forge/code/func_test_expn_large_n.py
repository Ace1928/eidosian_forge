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
def test_expn_large_n():
    dataset = []
    for n in [50, 51]:
        for x in np.logspace(0, 4, 200):
            with mpmath.workdps(100):
                dataset.append((n, x, float(mpmath.expint(n, x))))
    dataset = np.asarray(dataset)
    FuncData(sc.expn, dataset, (0, 1), 2, rtol=1e-13).check()