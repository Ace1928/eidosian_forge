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
def test_aliasing_b_ub(self):
    c = np.array([1.0])
    A_ub = np.array([[1.0]])
    b_ub_orig = np.array([3.0])
    b_ub = b_ub_orig.copy()
    bounds = (-4.0, np.inf)
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_fun=-4, desired_x=[-4])
    assert_allclose(b_ub_orig, b_ub)