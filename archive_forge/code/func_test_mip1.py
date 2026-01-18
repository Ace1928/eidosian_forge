import sys
import platform
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.optimize import linprog, OptimizeWarning
from scipy.optimize._numdiff import approx_derivative
from scipy.sparse.linalg import MatrixRankWarning
from scipy.linalg import LinAlgWarning
from scipy._lib._util import VisibleDeprecationWarning
import scipy.sparse
import pytest
@pytest.mark.xfail(condition=sys.maxsize < 2 ** 32 and platform.system() == 'Linux', run=False, reason='gh-16347')
def test_mip1(self):
    n = 4
    A, b, c, numbers, M = magic_square(n)
    bounds = [(0, 1)] * len(c)
    integrality = [1] * len(c)
    res = linprog(c=c * 0, A_eq=A, b_eq=b, bounds=bounds, method=self.method, integrality=integrality)
    s = (numbers.flatten() * res.x).reshape(n ** 2, n, n)
    square = np.sum(s, axis=0)
    np.testing.assert_allclose(square.sum(axis=0), M)
    np.testing.assert_allclose(square.sum(axis=1), M)
    np.testing.assert_allclose(np.diag(square).sum(), M)
    np.testing.assert_allclose(np.diag(square[:, ::-1]).sum(), M)
    np.testing.assert_allclose(res.x, np.round(res.x), atol=1e-12)