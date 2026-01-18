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
@pytest.mark.parametrize('options', [{'maxiter': -1}, {'disp': -1}, {'presolve': -1}, {'time_limit': -1}, {'dual_feasibility_tolerance': -1}, {'primal_feasibility_tolerance': -1}, {'ipm_optimality_tolerance': -1}, {'simplex_dual_edge_weight_strategy': 'ekki'}])
def test_invalid_option_values(self, options):

    def f(options):
        linprog(1, method=self.method, options=options)
    options.update(self.options)
    assert_warns(OptimizeWarning, f, options=options)