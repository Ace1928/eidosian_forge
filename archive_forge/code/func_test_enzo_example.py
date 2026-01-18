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
def test_enzo_example(self):
    c = [4, 8, 3, 0, 0, 0]
    A_eq = [[2, 5, 3, -1, 0, 0], [3, 2.5, 8, 0, -1, 0], [8, 10, 4, 0, 0, -1]]
    b_eq = [185, 155, 600]
    res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    _assert_success(res, desired_fun=317.5, desired_x=[66.25, 0, 17.5, 0, 183.75, 0], atol=6e-06, rtol=1e-07)