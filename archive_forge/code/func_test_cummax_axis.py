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
def test_cummax_axis(self) -> None:
    """Test the cumsum axis bug with row or column matrix
           See issue #1678
        """
    n = 5
    x1 = cp.Variable((1, n))
    expr1 = cp.cummax(x1, axis=0)
    prob1 = cp.Problem(cp.Maximize(cp.sum(x1)), [expr1 <= 1])
    prob1.solve()
    expect = np.ones((1, n))
    self.assertItemsAlmostEqual(expr1.value, expect)
    x2 = cp.Variable((n, 1))
    expr2 = cp.cummax(x2, axis=1)
    prob2 = cp.Problem(cp.Maximize(cp.sum(x2)), [expr2 <= 1])
    prob2.solve()
    expect = np.ones((n, 1))
    self.assertItemsAlmostEqual(expr2.value, expect)