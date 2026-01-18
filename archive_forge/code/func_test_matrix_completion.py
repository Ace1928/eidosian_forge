import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_matrix_completion(self) -> None:
    X = cp.Variable((3, 3), pos=True)
    obj = cp.Minimize(cp.sum(X))
    known_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1], [2, 0], [2, 1]]))
    known_values = np.array([1.0, 1.9, 0.8, 3.2, 5.9])
    param = cp.Parameter(shape=known_values.shape, pos=True, value=known_values)
    beta = cp.Parameter(pos=True, value=1.0)
    constr = [X[known_indices] == param, X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == beta]
    problem = cp.Problem(obj, constr)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.01)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.0001)