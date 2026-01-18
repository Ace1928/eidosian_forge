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
def test_bug_8174_2(self):
    c = np.array([1, 0, 0, 0, 0, 0, 0])
    A_ub = -np.identity(7)
    b_ub = np.array([[-2], [-2], [-2], [-2], [-2], [-2], [-2]])
    A_eq = np.array([[1, 1, 1, 1, 1, 1, 0], [0.3, 1.3, 0.9, 0, 0, 0, -1], [0.3, 0, 0, 0, 0, 0, -2 / 3], [0, 0.65, 0, 0, 0, 0, -1 / 15], [0, 0, 0.3, 0, 0, 0, -1 / 15]])
    b_eq = np.array([[100], [0], [0], [0], [0]])
    with suppress_warnings() as sup:
        if has_umfpack:
            sup.filter(UmfpackWarning)
        sup.filter(OptimizeWarning, 'A_eq does not appear...')
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_fun=43.3333333331385)