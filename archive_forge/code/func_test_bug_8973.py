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
def test_bug_8973(self):
    """
        Test whether bug described at:
        https://github.com/scipy/scipy/issues/8973
        was fixed.
        """
    c = np.array([0, 0, 0, 1, -1])
    A_ub = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    b_ub = np.array([2, -2])
    bounds = [(None, None), (None, None), (None, None), (-1, 1), (-1, 1)]
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_fun=-2)
    assert_equal(c @ res.x, res.fun)