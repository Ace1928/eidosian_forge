import os
import tempfile
from pulp.constants import PulpError
from pulp.apis import *
from pulp import LpVariable, LpProblem, lpSum, LpConstraintVar, LpFractionConstraint
from pulp import constants as const
from pulp.tests.bin_packing_problem import create_bin_packing_problem
from pulp.utilities import makeDict
import functools
import unittest
def test_infeasible_problem__is_not_valid(self):
    """Given a problem where x cannot converge to any value
            given conflicting constraints, assert that it is invalid."""
    name = self._testMethodName
    prob = LpProblem(name, const.LpMaximize)
    x = LpVariable('x')
    prob += 1 * x
    prob += x >= 2
    prob += x <= 1
    if self.solver.name in ['GUROBI_CMD', 'FSCIP_CMD']:
        pulpTestCheck(prob, self.solver, [const.LpStatusNotSolved, const.LpStatusInfeasible, const.LpStatusUndefined])
    else:
        pulpTestCheck(prob, self.solver, [const.LpStatusInfeasible, const.LpStatusUndefined])
    self.assertFalse(prob.valid())