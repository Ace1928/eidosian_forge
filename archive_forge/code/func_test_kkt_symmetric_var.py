import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
def test_kkt_symmetric_var(self, places=4):
    sth = TestKKT_Flags.symmetric_flag()
    sth.solve(solver='SCS')
    sth.check_primal_feasibility(places)
    sth.check_complementarity(places)
    sth.check_dual_domains(places)
    sth.check_stationary_lagrangian(places)
    return sth