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
def test_bug_10124(self):
    """
        Test for linprog docstring problem
        'disp'=True caused revised simplex failure
        """
    c = np.zeros(1)
    A_ub = np.array([[1]])
    b_ub = np.array([-2])
    bounds = (None, None)
    c = [-1, 4]
    A_ub = [[-3, 1], [1, 2]]
    b_ub = [6, 4]
    bounds = [(None, None), (-3, None)]
    o = {'disp': True}
    o.update(self.options)
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
    _assert_success(res, desired_x=[10, -3], desired_fun=-22)