import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_eps_keyword(self) -> None:
    """Test that the eps keyword is accepted"""
    x = cp.Variable()
    prob = cp.Problem(cp.Minimize(x), [x >= 0])
    prob.solve(solver=cp.MOSEK, eps=1e-08, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-06})
    assert prob.status is cp.OPTIMAL
    import mosek
    with pytest.raises(mosek.Error, match='The parameter value 0.1 is too large'):
        prob.solve(solver=cp.MOSEK, eps=0.1, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-06})
    from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK
    all_params = MOSEK.tolerance_params()
    prob.solve(solver=cp.MOSEK, eps=0.1, mosek_params={p: 1e-06 for p in all_params})
    assert prob.status is cp.OPTIMAL
    with pytest.raises(AssertionError, match='not compatible'):
        prob.solve(solver=cp.MOSEK, eps=0.1, mosek_params={mosek.dparam.intpnt_co_tol_dfeas: 1e-06})