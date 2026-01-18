from typing import Tuple
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def symvar_kronl(self, param):
    X = cp.Variable(shape=(2, 2), symmetric=True)
    b_val = 1.5 * np.ones((1, 1))
    if param:
        b = cp.Parameter(shape=(1, 1))
        b.value = b_val
    else:
        b = cp.Constant(b_val)
    L = np.array([[0.5, 1], [2, 3]])
    U = np.array([[10, 11], [12, 13]])
    kronX = cp.kron(X, b)
    objective = cp.Minimize(cp.sum(X.flatten()))
    constraints = [U >= kronX, kronX >= L]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    self.assertItemsAlmostEqual(X.value, np.array([[0.5, 2], [2, 3]]) / 1.5)
    objective = cp.Maximize(cp.sum(X.flatten()))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    self.assertItemsAlmostEqual(X.value, np.array([[10, 11], [11, 13]]) / 1.5)
    pass