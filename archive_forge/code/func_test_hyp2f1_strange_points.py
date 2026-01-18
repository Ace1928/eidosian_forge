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
@check_version(mpmath, '1.0.0')
def test_hyp2f1_strange_points():
    pts = [(2, -1, -1, 0.7), (2, -2, -2, 0.7)]
    pts += list(itertools.product([2, 1, -0.7, -1000], repeat=4))
    pts = [(a, b, c, x) for a, b, c, x in pts if b == c and round(b) == b and (b < 0) and (b != -1000)]
    kw = dict(eliminate=True)
    dataset = [p + (float(mpmath.hyp2f1(*p, **kw)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float64)
    FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()