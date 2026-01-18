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
def test_magic_square_sparse_no_presolve(self):
    A_eq, b_eq, c, _, _ = magic_square(3)
    bounds = (0, 1)
    with suppress_warnings() as sup:
        if has_umfpack:
            sup.filter(UmfpackWarning)
        sup.filter(MatrixRankWarning, 'Matrix is exactly singular')
        sup.filter(OptimizeWarning, 'Solving system with option...')
        o = {key: self.options[key] for key in self.options}
        o['presolve'] = False
        res = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds, method=self.method, options=o)
    _assert_success(res, desired_fun=1.730550597)