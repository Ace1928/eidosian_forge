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
def test_mosek_accept_unknown(self) -> None:
    mosek_param = {'MSK_IPAR_INTPNT_MAX_ITERATIONS': 0}
    sth = sths.lp_5()
    sth.solve(solver=cp.MOSEK, accept_unknown=True, mosek_params=mosek_param)
    assert sth.prob.status in {cp.OPTIMAL_INACCURATE, cp.OPTIMAL}
    with pytest.raises(cp.error.SolverError, match="Solver 'MOSEK' failed"):
        sth.solve(solver=cp.MOSEK, mosek_params=mosek_param)