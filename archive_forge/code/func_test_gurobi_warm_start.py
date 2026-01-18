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
def test_gurobi_warm_start(self) -> None:
    """Make sure that warm starting Gurobi behaves as expected
           Note: This only checks output, not whether or not Gurobi is warm starting internally
        """
    if cp.GUROBI in INSTALLED_SOLVERS:
        import gurobipy
        import numpy as np
        A = cp.Parameter((2, 2))
        b = cp.Parameter(2)
        h = cp.Parameter(2)
        c = cp.Parameter(2)
        A.value = np.array([[1, 0], [0, 0]])
        b.value = np.array([1, 0])
        h.value = np.array([2, 2])
        c.value = np.array([1, 1])
        objective = cp.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
        constraints = [self.x[0] ** 2 <= h[0] ** 2, self.x[1] <= h[1], A @ self.x == b]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.GUROBI, warm_start=True)
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])
        A.value = np.array([[0, 0], [0, 1]])
        b.value = np.array([0, 1])
        h.value = np.array([2, 2])
        c.value = np.array([1, 1])
        result = prob.solve(solver=cp.GUROBI, warm_start=True)
        self.assertAlmostEqual(result, 3)
        self.assertItemsAlmostEqual(self.x.value, [2, 1])
        A.value = np.array([[1, 0], [0, 0]])
        b.value = np.array([1, 0])
        h.value = np.array([1, 1])
        c.value = np.array([1, 1])
        result = prob.solve(solver=cp.GUROBI, warm_start=True)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])
        A.value = np.array([[1, 0], [0, 0]])
        b.value = np.array([1, 0])
        h.value = np.array([2, 2])
        c.value = np.array([2, 1])
        result = prob.solve(solver=cp.GUROBI, warm_start=True)
        self.assertEqual(result, 4)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])
        init_value = np.array([2, 3])
        self.x.value = init_value
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.GUROBI, warm_start=True)
        self.assertEqual(result, 4)
        self.assertItemsAlmostEqual(self.x.value, [1, 2])
        model = prob.solver_stats.extra_stats
        model_x = model.getVars()
        for i in range(self.x.size):
            assert init_value[i] == model_x[i].start
            assert np.isclose(self.x.value[i], model_x[i].x)
        z = cp.Variable()
        Y = cp.Variable((3, 2))
        Y_val = np.reshape(np.arange(6), (3, 2))
        Y.value = Y_val + 1
        objective = cp.Maximize(z + cp.sum(Y))
        constraints = [Y <= Y_val, z <= 2]
        prob = cp.Problem(objective, constraints)
        result = prob.solve(solver=cp.GUROBI, warm_start=True)
        self.assertEqual(result, Y_val.sum() + 2)
        self.assertAlmostEqual(z.value, 2)
        self.assertItemsAlmostEqual(Y.value, Y_val)
        model = prob.solver_stats.extra_stats
        model_x = model.getVars()
        assert gurobipy.GRB.UNDEFINED == model_x[0].start
        assert np.isclose(2, model_x[0].x)
        for i in range(1, Y.size + 1):
            row = (i - 1) % Y.shape[0]
            col = (i - 1) // Y.shape[0]
            assert Y_val[row, col] + 1 == model_x[i].start
            assert np.isclose(Y.value[row, col], model_x[i].x)
    else:
        with self.assertRaises(Exception) as cm:
            prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
            prob.solve(solver=cp.GUROBI, warm_start=True)
        self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % cp.GUROBI)