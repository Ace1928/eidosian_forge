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
def test_bug_8664(self):
    c = [4]
    A_ub = [[2], [5]]
    b_ub = [4, 4]
    A_eq = [[0], [-8], [9]]
    b_eq = [3, 2, 10]
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        sup.filter(OptimizeWarning, 'Solving system with option...')
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options={'presolve': False})
    assert_(not res.success, 'Incorrectly reported success')