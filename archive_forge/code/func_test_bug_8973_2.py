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
def test_bug_8973_2(self):
    """
        Additional test for:
        https://github.com/scipy/scipy/issues/8973
        suggested in
        https://github.com/scipy/scipy/pull/8985
        review by @antonior92
        """
    c = np.zeros(1)
    A_ub = np.array([[1]])
    b_ub = np.array([-2])
    bounds = (None, None)
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_x=[-2], desired_fun=0)