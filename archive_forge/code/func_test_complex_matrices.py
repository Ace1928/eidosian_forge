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
def test_complex_matrices(self) -> None:
    """Test complex matrices.
        """
    np.random.seed(0)
    K = np.array(np.random.rand(2, 2) + 1j * np.random.rand(2, 2))
    n1 = la.svdvals(K).sum()
    X = cp.Variable((2, 2), complex=True)
    Y = cp.Variable((2, 2), complex=True)
    objective = cp.Minimize(cp.real(0.5 * cp.trace(X) + 0.5 * cp.trace(Y)))
    constraints = [cp.bmat([[X, -K.conj().T], [-K, Y]]) >> 0, X >> 0, Y >> 0]
    problem = cp.Problem(objective, constraints)
    sol_scs = problem.solve(solver='SCS')
    self.assertEqual(constraints[0].dual_value.shape, (4, 4))
    self.assertEqual(constraints[1].dual_value.shape, (2, 2))
    self.assertEqual(constraints[2].dual_value.shape, (2, 2))
    self.assertAlmostEqual(sol_scs, n1)