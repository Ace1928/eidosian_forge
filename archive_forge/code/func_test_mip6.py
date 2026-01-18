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
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_mip6(self):
    A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26], [39, 16, 22, 28, 26, 30, 23, 24], [18, 14, 29, 27, 30, 38, 26, 26], [41, 26, 28, 36, 18, 38, 16, 26]])
    b_eq = np.array([7872, 10466, 11322, 12058])
    c = np.array([2, 10, 13, 17, 7, 5, 7, 3])
    bounds = [(0, np.inf)] * 8
    integrality = [1] * 8
    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method=self.method, integrality=integrality)
    np.testing.assert_allclose(res.fun, 1854)