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
def test_bug_8561(self):
    c = np.array([7, 0, -4, 1.5, 1.5])
    A_ub = np.array([[4, 5.5, 1.5, 1.0, -3.5], [1, -2.5, -2, 2.5, 0.5], [3, -0.5, 4, -12.5, -7], [-1, 4.5, 2, -3.5, -2], [5.5, 2, -4.5, -1, 9.5]])
    b_ub = np.array([0, 0, 0, 0, 1])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, options=self.options, method=self.method)
    _assert_success(res, desired_x=[0, 0, 19, 16 / 3, 29 / 3])