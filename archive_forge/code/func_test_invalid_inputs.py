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
def test_invalid_inputs(self):

    def f(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None):
        linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=self.options)
    assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, 2), (3, 4)])
    with np.testing.suppress_warnings() as sup:
        sup.filter(VisibleDeprecationWarning, 'Creating an ndarray from ragged')
        assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, 2), (3, 4), (3, 4, 5)])
    assert_raises(ValueError, f, [1, 2, 3], bounds=[(1, -2), (1, 2)])
    assert_raises(ValueError, f, [1, 2], A_ub=[[1, 2]], b_ub=[1, 2])
    assert_raises(ValueError, f, [1, 2], A_ub=[[1]], b_ub=[1])
    assert_raises(ValueError, f, [1, 2], A_eq=[[1, 2]], b_eq=[1, 2])
    assert_raises(ValueError, f, [1, 2], A_eq=[[1]], b_eq=[1])
    assert_raises(ValueError, f, [1, 2], A_eq=[1], b_eq=1)
    if '_sparse_presolve' in self.options and self.options['_sparse_presolve']:
        return
    assert_raises(ValueError, f, [1, 2], A_ub=np.zeros((1, 1, 3)), b_eq=1)