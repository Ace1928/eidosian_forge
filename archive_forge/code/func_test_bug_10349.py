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
def test_bug_10349(self):
    """
        Test for redundancy removal tolerance issue
        https://github.com/scipy/scipy/issues/10349
        """
    A_eq = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 0, 1]])
    b_eq = np.array([221, 210, 10, 141, 198, 102])
    c = np.concatenate((0, 1, np.zeros(4)), axis=None)
    with suppress_warnings() as sup:
        sup.filter(OptimizeWarning, 'A_eq does not appear...')
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_x=[129, 92, 12, 198, 0, 10], desired_fun=92)