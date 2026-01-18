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
def test_huber_scs(self) -> None:
    """Test that huber regression works with SCS.
           See issue #1370.
        """
    np.random.seed(1)
    m = 5
    n = 2
    x0 = np.random.randn(n)
    A = np.random.randn(m, n)
    b = A.dot(x0) + 0.01 * np.random.randn(m)
    k = int(0.02 * m)
    idx = np.random.randint(m, size=k)
    b[idx] += 10 * np.random.randn(k)
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(A @ x - b))))
    prob.solve(solver=cp.SCS)