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
def test_bounds_infeasible_2(self):
    do_presolve = self.options.get('presolve', True)
    simplex_without_presolve = not do_presolve and self.method == 'simplex'
    c = [1, 2, 3]
    bounds_1 = [(1, 2), (np.inf, np.inf), (3, 4)]
    bounds_2 = [(1, 2), (-np.inf, -np.inf), (3, 4)]
    if simplex_without_presolve:

        def g(c, bounds):
            res = linprog(c, bounds=bounds, method=self.method, options=self.options)
            return res
        with pytest.warns(RuntimeWarning):
            with pytest.raises(IndexError):
                g(c, bounds=bounds_1)
        with pytest.warns(RuntimeWarning):
            with pytest.raises(IndexError):
                g(c, bounds=bounds_2)
    else:
        res = linprog(c=c, bounds=bounds_1, method=self.method, options=self.options)
        _assert_infeasible(res)
        if do_presolve:
            assert_equal(res.nit, 0)
        res = linprog(c=c, bounds=bounds_2, method=self.method, options=self.options)
        _assert_infeasible(res)
        if do_presolve:
            assert_equal(res.nit, 0)