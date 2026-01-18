import warnings
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
@staticmethod
def test_mi_lp_4(solver, places: int=4, **kwargs) -> SolverTestHelper:
    sth = mi_lp_4()
    sth.solve(solver, **kwargs)
    sth.verify_objective(places)
    sth.verify_primal_values(places)
    return sth