import numpy as np
import scipy as sp
import cvxpy as cp
from cvxpy import trace
from cvxpy.atoms import von_neumann_entr
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.utilities.linalg import onb_for_orthogonal_complement
def test_1_real(self):
    sth = Test_von_neumann_entr.make_test_1(False)
    sth.solve(**self.SOLVE_ARGS)
    sth.verify_objective(places=3)
    sth.verify_primal_values(places=3)