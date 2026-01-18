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
def test_mip2(self):
    A_ub = np.array([[2, -2], [-8, 10]])
    b_ub = np.array([-1, 13])
    c = -np.array([1, 1])
    bounds = np.array([(0, np.inf)] * len(c))
    integrality = np.ones_like(c)
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=self.method, integrality=integrality)
    np.testing.assert_allclose(res.x, [1, 2])
    np.testing.assert_allclose(res.fun, -3)