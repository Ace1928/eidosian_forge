import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def test_non_affine_exception(ineq_form: bool):
    x = cp.Variable()
    x_abs = cp.abs(x)
    set_vals = {1, 2, 3}
    with pytest.raises(ValueError, match='must be affine'):
        FiniteSet(x_abs, set_vals, ineq_form=ineq_form)