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
def test_empty_constraint_2(self):
    c = [-1, 1, -1, 1]
    bounds = [(0, np.inf), (-np.inf, 0), (-1, 1), (-1, 1)]
    res = linprog(c, bounds=bounds, method=self.method, options=self.options)
    _assert_unbounded(res)
    if self.options.get('presolve', True):
        assert_equal(res.nit, 0)