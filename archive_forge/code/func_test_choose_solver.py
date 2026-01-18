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
def test_choose_solver():
    c = np.array([-3, -2])
    A_ub = [[2, 1], [1, 1], [1, 0]]
    b_ub = [10, 8, 4]
    res = linprog(c, A_ub, b_ub, method='highs')
    _assert_success(res, desired_fun=-18.0, desired_x=[2, 6])