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
def test_get_problem_data(self) -> None:
    """Test get_problem_data method.
        """
    data, _, _ = Problem(cp.Minimize(cp.exp(self.a) + 2)).get_problem_data(s.SCS)
    dims = data[ConicSolver.DIMS]
    self.assertEqual(dims.exp, 1)
    self.assertEqual(data['c'].shape, (2,))
    self.assertEqual(data['A'].shape, (3, 2))
    data, _, _ = Problem(cp.Minimize(cp.norm(self.x) + 3)).get_problem_data(s.ECOS)
    dims = data[ConicSolver.DIMS]
    self.assertEqual(dims.soc, [3])
    self.assertEqual(data['c'].shape, (3,))
    self.assertIsNone(data['A'])
    self.assertEqual(data['G'].shape, (3, 3))
    p = Problem(cp.Minimize(cp.sum_squares(self.x) + 2))
    data, _, _ = p.get_problem_data(s.SCS, solver_opts={'use_quad_obj': False})
    dims = data[ConicSolver.DIMS]
    self.assertEqual(dims.soc, [4])
    self.assertEqual(data['c'].shape, (3,))
    self.assertEqual(data['A'].shape, (4, 3))
    data, _, _ = p.get_problem_data(s.SCS, solver_opts={'use_quad_obj': True})
    dims = data[ConicSolver.DIMS]
    self.assertEqual(dims.soc, [])
    self.assertEqual(data['P'].shape, (2, 2))
    self.assertEqual(data['c'].shape, (2,))
    self.assertEqual(data['A'].shape, (0, 2))
    if s.CVXOPT in INSTALLED_SOLVERS:
        data, _, _ = Problem(cp.Minimize(cp.norm(self.x) + 3)).get_problem_data(s.CVXOPT)
        dims = data[ConicSolver.DIMS]
        self.assertEqual(dims.soc, [3])