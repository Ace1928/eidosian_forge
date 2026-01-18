from __future__ import division
import numpy as np
import cvxpy as cp
from cvxpy import Maximize, Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms import linearize
from cvxpy.transforms.partial_optimize import partial_optimize
def test_affine(self) -> None:
    """Test grad for affine atoms.
        """
    expr = -self.a
    self.a.value = 2
    self.assertAlmostEqual(expr.grad[self.a], -1)
    expr = 2 * self.a
    self.a.value = 2
    self.assertAlmostEqual(expr.grad[self.a], 2)
    expr = self.a / 2
    self.a.value = 2
    self.assertAlmostEqual(expr.grad[self.a], 0.5)
    expr = -self.x
    self.x.value = [3, 4]
    val = np.zeros((2, 2)) - np.diag([1, 1])
    self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
    expr = -self.A
    self.A.value = [[1, 2], [3, 4]]
    val = np.zeros((4, 4)) - np.diag([1, 1, 1, 1])
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)
    expr = self.A[0, 1]
    self.A.value = [[1, 2], [3, 4]]
    val = np.zeros((4, 1))
    val[2] = 1
    self.assertItemsAlmostEqual(expr.grad[self.A].toarray(), val)
    z = Variable(3)
    expr = cp.hstack([self.x, z])
    self.x.value = [1, 2]
    z.value = [1, 2, 3]
    val = np.zeros((2, 5))
    val[:, 0:2] = np.eye(2)
    self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
    val = np.zeros((3, 5))
    val[:, 2:] = np.eye(3)
    self.assertItemsAlmostEqual(expr.grad[z].toarray(), val)
    expr = cp.cumsum(self.x)
    self.x.value = [1, 2]
    val = np.ones((2, 2))
    val[1, 0] = 0
    self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)
    expr = cp.cumsum(self.x[:, None], axis=1)
    self.x.value = [1, 2]
    val = np.eye(2)
    self.assertItemsAlmostEqual(expr.grad[self.x].toarray(), val)