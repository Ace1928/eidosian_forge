from typing import Tuple
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def scalar_kronl(self, param):
    y = cp.Variable(shape=(1, 1))
    A_val = np.array([[1.0, 2.0], [3.0, 4.0]])
    L = np.array([[0.5, 1], [2, 3]])
    U = np.array([[10, 11], [12, 13]])
    if param:
        A = cp.Parameter(shape=(2, 2))
        A.value = A_val
    else:
        A = cp.Constant(A_val)
    krony = cp.kron(y, A)
    constraints = [U >= krony, krony >= L]
    objective = cp.Minimize(y)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    self.assertItemsAlmostEqual(y.value, np.array([[np.max(L / A_val)]]))
    objective = cp.Maximize(y)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    self.assertItemsAlmostEqual(y.value, np.array([[np.min(U / A_val)]]))
    pass