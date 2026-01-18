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
def test_installed_solvers(self) -> None:
    """Test the list of installed solvers.
        """
    from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, SOLVER_MAP_CONIC, SOLVER_MAP_QP
    prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1) + 1.0), [self.x == 0])
    for solver in SOLVER_MAP_CONIC.keys():
        if solver in INSTALLED_SOLVERS:
            prob.solve(solver=solver)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])
        else:
            with self.assertRaises(Exception) as cm:
                prob.solve(solver=solver)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % solver)
    for solver in SOLVER_MAP_QP.keys():
        if solver in INSTALLED_SOLVERS:
            prob.solve(solver=solver)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])
        else:
            with self.assertRaises(Exception) as cm:
                prob.solve(solver=solver)
            self.assertEqual(str(cm.exception), 'The solver %s is not installed.' % solver)