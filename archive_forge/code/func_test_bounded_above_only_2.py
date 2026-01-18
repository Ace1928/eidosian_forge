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
def test_bounded_above_only_2(self):
    c = np.ones(3)
    A_eq = np.eye(3)
    b_eq = np.array([1, 2, 3])
    bounds = (-np.inf, 4)
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_x=b_eq, desired_fun=np.sum(b_eq))