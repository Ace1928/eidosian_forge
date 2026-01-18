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
def test_bug_7044(self):
    A_eq, b_eq, c, _, _ = magic_square(3)
    with suppress_warnings() as sup:
        sup.filter(OptimizeWarning, 'A_eq does not appear...')
        sup.filter(RuntimeWarning, 'invalid value encountered')
        sup.filter(LinAlgWarning)
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    desired_fun = 1.730550597
    _assert_success(res, desired_fun=desired_fun)
    assert_allclose(A_eq.dot(res.x), b_eq)
    assert_array_less(np.zeros(res.x.size) - 1e-05, res.x)