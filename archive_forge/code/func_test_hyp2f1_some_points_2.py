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
@check_version(mpmath, '0.14')
def test_hyp2f1_some_points_2():
    pts = [(112, (51, 10), (-9, 10), -0.99999), (10, -900, 10.5, 0.99), (10, -900, -10.5, 0.99)]

    def fev(x):
        if isinstance(x, tuple):
            return float(x[0]) / x[1]
        else:
            return x
    dataset = [tuple(map(fev, p)) + (float(mpmath.hyp2f1(*p)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float64)
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()