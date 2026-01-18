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
@gurobi_test
def test_measuring_solving_time(self):
    print('\t Testing measuring optimization time')
    time_limit = 10
    solver_settings = dict(PULP_CBC_CMD=30, COIN_CMD=30, SCIP_CMD=30, GUROBI_CMD=50, CPLEX_CMD=50, GUROBI=50, HiGHS=50)
    bins = solver_settings.get(self.solver.name)
    if bins is None:
        return
    prob = create_bin_packing_problem(bins=bins, seed=99)
    self.solver.timeLimit = time_limit
    status = prob.solve(self.solver)
    delta = 20
    reported_time = prob.solutionTime
    if self.solver.name in ['PULP_CBC_CMD', 'COIN_CMD']:
        reported_time = prob.solutionCpuTime
    self.assertAlmostEqual(reported_time, time_limit, delta=delta, msg=f'optimization time for solver {self.solver.name}')
    self.assertTrue(prob.objective.value() is not None)
    self.assertEqual(status, const.LpStatusOptimal)
    for v in prob.variables():
        self.assertTrue(v.varValue is not None)