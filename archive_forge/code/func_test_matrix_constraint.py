import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_matrix_constraint(self) -> None:
    X = cp.Variable((2, 2), pos=True)
    a = cp.Parameter(pos=True, value=0.1)
    obj = cp.Minimize(cp.geo_mean(cp.vec(X)))
    constr = [cp.diag(X) == a, cp.hstack([X[0, 1], X[1, 0]]) == 2 * a]
    problem = cp.Problem(obj, constr)
    gradcheck(problem, gp=True)
    perturbcheck(problem, gp=True)