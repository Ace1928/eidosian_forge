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
@check_version(mpmath, '0.13')
def test_hyp2f1_real_some_points():
    pts = [(1, 2, 3, 0), (1.0 / 3, 2.0 / 3, 5.0 / 6, 27.0 / 32), (1.0 / 4, 1.0 / 2, 3.0 / 4, 80.0 / 81), (2, -2, -3, 3), (2, -3, -2, 3), (2, -1.5, -1.5, 3), (1, 2, 3, 0), (0.7235, -1, -5, 0.3), (0.25, 1.0 / 3, 2, 0.999), (0.25, 1.0 / 3, 2, -1), (2, 3, 5, 0.99), (3.0 / 2, -0.5, 3, 0.99), (2, 2.5, -3.25, 0.999), (-8, 18.016500331508873, 10.805295997850628, 0.90875647507), (-10, 900, -10.5, 0.99), (-10, 900, 10.5, 0.99), (-1, 2, 1, 1.0), (-1, 2, 1, -1.0), (-3, 13, 5, 1.0), (-3, 13, 5, -1.0), (0.5, 1 - 270.5, 1.5, 0.999 ** 2)]
    dataset = [p + (float(mpmath.hyp2f1(*p)),) for p in pts]
    dataset = np.array(dataset, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        FuncData(sc.hyp2f1, dataset, (0, 1, 2, 3), 4, rtol=1e-10).check()