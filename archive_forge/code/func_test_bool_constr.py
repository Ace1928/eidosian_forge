import builtins
import pickle
import sys
import warnings
from fractions import Fraction
from io import StringIO
import ecos
import numpy
import numpy as np
import scipy.sparse as sp
import scs
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import PSD, ExpCone, NonNeg, Zero
from cvxpy.error import DCPError, ParameterError, SolverError
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.conic_solvers import ecos_conif, scs_conif
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import (
from cvxpy.reductions.solvers.solving_chain import ECOS_DEPRECATION_MSG
from cvxpy.tests.base_test import BaseTest
def test_bool_constr(self) -> None:
    """Test constraints that evaluate to booleans.
        """
    x = cp.Variable(pos=True)
    prob = cp.Problem(cp.Minimize(x), [True])
    prob.solve(solver=cp.ECOS)
    self.assertAlmostEqual(x.value, 0)
    x = cp.Variable(pos=True)
    prob = cp.Problem(cp.Minimize(x), [True] * 10)
    prob.solve(solver=cp.ECOS)
    self.assertAlmostEqual(x.value, 0)
    prob = cp.Problem(cp.Minimize(x), [42 <= x] + [True] * 10)
    prob.solve(solver=cp.ECOS)
    self.assertAlmostEqual(x.value, 42)
    prob = cp.Problem(cp.Minimize(x), [True] + [42 <= x] + [True] * 10)
    prob.solve(solver=cp.ECOS)
    self.assertAlmostEqual(x.value, 42)
    prob = cp.Problem(cp.Minimize(x), [False])
    prob.solve(solver=cp.ECOS)
    self.assertEqual(prob.status, s.INFEASIBLE)
    prob = cp.Problem(cp.Minimize(x), [False] * 10)
    prob.solve(solver=cp.ECOS)
    self.assertEqual(prob.status, s.INFEASIBLE)
    prob = cp.Problem(cp.Minimize(x), [True] * 10 + [False] + [True] * 10)
    prob.solve(solver=cp.ECOS)
    self.assertEqual(prob.status, s.INFEASIBLE)
    prob = cp.Problem(cp.Minimize(x), [42 <= x] + [True] * 10 + [False])
    prob.solve(solver=cp.ECOS)
    self.assertEqual(prob.status, s.INFEASIBLE)
    prob = cp.Problem(cp.Minimize(x), [True] + [x <= -42] + [True] * 10)
    prob.solve(solver=cp.ECOS)
    self.assertEqual(prob.status, s.INFEASIBLE)