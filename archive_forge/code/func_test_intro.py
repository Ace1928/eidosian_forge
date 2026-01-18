from __future__ import print_function
import unittest
import numpy as np
import cvxpy as cvx
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest
def test_intro(self) -> None:
    """Test examples from cvxpy.org introduction.
        """
    import numpy
    m = 30
    n = 20
    numpy.random.seed(1)
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)
    x = cvx.Variable(n)
    objective = cvx.Minimize(cvx.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=cvx.SCS, eps=1e-06)
    print(x.value)
    print(constraints[0].dual_value)
    x = cvx.Variable()
    y = cvx.Variable()
    constraints = [x + y == 1, x - y >= 1]
    obj = cvx.Minimize(cvx.square(x - y))
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.SCS, eps=1e-06)
    print('status:', prob.status)
    print('optimal value', prob.value)
    print('optimal var', x.value, y.value)
    x = cvx.Variable()
    y = cvx.Variable()
    constraints = [x + y == 1, x - y >= 1]
    obj = cvx.Minimize(cvx.square(x - y))
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.SCS, eps=1e-06)
    print('status:', prob.status)
    print('optimal value', prob.value)
    print('optimal var', x.value, y.value)
    self.assertEqual(prob.status, cvx.OPTIMAL)
    self.assertAlmostEqual(prob.value, 1.0)
    self.assertAlmostEqual(x.value, 1.0)
    self.assertAlmostEqual(y.value, 0)
    prob = cvx.Problem(cvx.Maximize(x + y), prob.constraints)
    print('optimal value', prob.solve(solver=cvx.SCS, eps=1e-06))
    self.assertAlmostEqual(prob.value, 1.0, places=3)
    constraints = prob.constraints
    constraints[0] = x + y <= 3
    prob = cvx.Problem(prob.objective, constraints)
    print('optimal value', prob.solve(solver=cvx.SCS, eps=1e-06))
    self.assertAlmostEqual(prob.value, 3.0, places=2)
    x = cvx.Variable()
    prob = cvx.Problem(cvx.Minimize(x), [x >= 1, x <= 0])
    prob.solve(solver=cvx.SCS, eps=1e-06)
    print('status:', prob.status)
    print('optimal value', prob.value)
    self.assertEqual(prob.status, cvx.INFEASIBLE)
    self.assertAlmostEqual(prob.value, np.inf)
    prob = cvx.Problem(cvx.Minimize(x))
    prob.solve(solver=cvx.ECOS)
    print('status:', prob.status)
    print('optimal value', prob.value)
    self.assertEqual(prob.status, cvx.UNBOUNDED)
    self.assertAlmostEqual(prob.value, -np.inf)
    cvx.Variable()
    x = cvx.Variable(5)
    A = cvx.Variable((4, 7))
    import numpy
    m = 10
    n = 5
    numpy.random.seed(1)
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)
    x = cvx.Variable(n)
    objective = cvx.Minimize(cvx.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    prob = cvx.Problem(objective, constraints)
    print('Optimal value', prob.solve(solver=cvx.SCS, eps=1e-06))
    print('Optimal var')
    print(x.value)
    self.assertAlmostEqual(prob.value, 4.14133859146)
    m = cvx.Parameter(nonneg=True)
    cvx.Parameter(5)
    G = cvx.Parameter((4, 7), nonpos=True)
    G.value = -numpy.ones((4, 7))
    rho = cvx.Parameter(nonneg=True)
    rho.value = 2
    rho = cvx.Parameter(nonneg=True, value=2)
    import numpy
    n = 15
    m = 10
    numpy.random.seed(1)
    A = numpy.random.randn(n, m)
    b = numpy.random.randn(n)
    gamma = cvx.Parameter(nonneg=True)
    x = cvx.Variable(m)
    error = cvx.sum_squares(A @ x - b)
    obj = cvx.Minimize(error + gamma * cvx.norm(x, 1))
    prob = cvx.Problem(obj)
    sq_penalty = []
    l1_penalty = []
    x_values = []
    gamma_vals = numpy.logspace(-4, 6)
    for val in gamma_vals:
        gamma.value = val
        prob.solve(solver=cvx.SCS, eps=1e-06)
        sq_penalty.append(error.value)
        l1_penalty.append(cvx.norm(x, 1).value)
        x_values.append(x.value)
    import numpy
    X = cvx.Variable((5, 4))
    A = numpy.ones((3, 5))
    print('dimensions of X:', X.size)
    print('dimensions of sum(X):', cvx.sum(X).size)
    print('dimensions of A @ X:', (A @ X).size)
    try:
        A + X
    except ValueError as e:
        print(e)