import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_affine_atoms_canon(self) -> None:
    """Test canonicalization for affine atoms.
        """
    x = Variable()
    expr = cp.imag(x + 1j * x)
    prob = Problem(Minimize(expr), [x >= 0])
    result = prob.solve(solver='SCS', eps=1e-06)
    self.assertAlmostEqual(result, 0)
    self.assertAlmostEqual(x.value, 0)
    x = Variable(imag=True)
    expr = 1j * x
    prob = Problem(Minimize(expr), [cp.imag(x) <= 1])
    result = prob.solve(solver='SCS', eps=1e-06)
    self.assertAlmostEqual(result, -1)
    self.assertAlmostEqual(x.value, 1j)
    x = Variable(2)
    expr = x * 1j
    prob = Problem(Minimize(expr[0] * 1j + expr[1] * 1j), [cp.real(x + 1j) >= 1])
    result = prob.solve(solver='SCS', eps=1e-06)
    self.assertAlmostEqual(result, -np.inf)
    prob = Problem(Minimize(expr[0] * 1j + expr[1] * 1j), [cp.real(x + 1j) <= 1])
    result = prob.solve(solver='SCS', eps=1e-06)
    self.assertAlmostEqual(result, -2)
    self.assertItemsAlmostEqual(x.value, [1, 1])
    prob = Problem(Minimize(expr[0] * 1j + expr[1] * 1j), [cp.real(x + 1j) >= 1, cp.conj(x) <= 0])
    result = prob.solve(solver='SCS', eps=1e-06)
    self.assertAlmostEqual(result, np.inf)
    x = Variable((2, 2))
    y = Variable((3, 2), complex=True)
    expr = cp.vstack([x, y])
    prob = Problem(Minimize(cp.sum(cp.imag(cp.conj(expr)))), [x == 0, cp.real(y) == 0, cp.imag(y) <= 1])
    result = prob.solve(solver='SCS', eps=1e-06)
    self.assertAlmostEqual(result, -6)
    self.assertItemsAlmostEqual(y.value, 1j * np.ones((3, 2)))
    self.assertItemsAlmostEqual(x.value, np.zeros((2, 2)))
    x = Variable((2, 2))
    y = Variable((3, 2), complex=True)
    expr = cp.vstack([x, y])
    prob = Problem(Minimize(cp.sum(cp.imag(expr.H))), [x == 0, cp.real(y) == 0, cp.imag(y) <= 1])
    result = prob.solve(solver='SCS', eps=1e-06)
    self.assertAlmostEqual(result, -6)
    self.assertItemsAlmostEqual(y.value, 1j * np.ones((3, 2)))
    self.assertItemsAlmostEqual(x.value, np.zeros((2, 2)))