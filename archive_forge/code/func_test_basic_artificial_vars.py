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
def test_basic_artificial_vars(self):
    c = np.array([-0.1, -0.07, 0.004, 0.004, 0.004, 0.004])
    A_ub = np.array([[1.0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0], [0, -1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0], [1.0, 1.0, 0, 0, 0, 0]])
    b_ub = np.array([3.0, 3.0, 3.0, 3.0, 20.0])
    A_eq = np.array([[1.0, 0, -1, 1, -1, 1], [0, -1.0, -1, 1, -1, 1]])
    b_eq = np.array([0, 0])
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_fun=0, desired_x=np.zeros_like(c), atol=2e-06)