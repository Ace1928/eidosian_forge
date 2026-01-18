import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
@staticmethod
def test_lp_6(solver, places: int=4, duals: bool=True, **kwargs) -> SolverTestHelper:
    sth = lp_6()
    sth.solve(solver, **kwargs)
    sth.verify_objective(places)
    sth.check_primal_feasibility(places)
    if duals:
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
    return sth