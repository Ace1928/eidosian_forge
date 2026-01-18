import os
import sys
from os.path import dirname, abspath
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, ProblemSense, TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.CBCplugin import CBCSHELL
@unittest.skipIf(not cbc_available, "The 'cbc' solver is not available")
def test_infeasible_mip(self):
    self.model.X = Var(within=NonNegativeIntegers)
    self.model.C1 = Constraint(expr=self.model.X <= 1)
    self.model.C2 = Constraint(expr=self.model.X >= 2)
    self.model.Obj = Objective(expr=self.model.X, sense=minimize)
    results = self.opt.solve(self.model)
    self.assertEqual(ProblemSense.minimize, results.problem.sense)
    self.assertEqual(TerminationCondition.infeasible, results.solver.termination_condition)
    self.assertEqual('Model was proven to be infeasible.', results.solver.termination_message)
    self.assertEqual(SolverStatus.warning, results.solver.status)