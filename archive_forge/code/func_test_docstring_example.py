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
def test_docstring_example(self):
    c = [-1, 4]
    A = [[-3, 1], [1, 2]]
    b = [6, 4]
    x0_bounds = (None, None)
    x1_bounds = (-3, None)
    res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds), options=self.options, method=self.method)
    _assert_success(res, desired_fun=-22)