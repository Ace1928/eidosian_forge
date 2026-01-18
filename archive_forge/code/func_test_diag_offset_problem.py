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
def test_diag_offset_problem(self) -> None:
    n = 4
    A = np.arange(int(n ** 2)).reshape((n, n))
    for k in range(-n + 1, n):
        x = cp.Variable(n - abs(k))
        obj = cp.Minimize(cp.sum(x))
        constraints = [cp.diag(x, k) == np.diag(np.diag(A, k), k)]
        prob = cp.Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, np.sum(np.diag(A, k)))
        assert np.allclose(x.value, np.diag(A, k), atol=0.0001)
        X = cp.Variable((n, n), nonneg=True)
        obj = cp.Minimize(cp.sum(X))
        constraints = [cp.diag(X, k) == np.diag(A, k)]
        prob = cp.Problem(obj, constraints)
        result = prob.solve(solver=cp.SCS, eps=1e-06)
        self.assertAlmostEqual(result, np.sum(np.diag(A, k)))
        assert np.allclose(X.value, np.diag(np.diag(A, k), k), atol=0.0001)