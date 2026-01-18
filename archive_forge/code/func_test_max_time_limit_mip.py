import os
import sys
from os.path import dirname, abspath
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, ProblemSense, TerminationCondition, SolverStatus
from pyomo.solvers.plugins.solvers.CBCplugin import CBCSHELL
def test_max_time_limit_mip(self):
    """
        solver_kwargs={'timelimit': 0.1}
        """
    lp_file = 'max_time_limit.out.lp'
    results = self.opt.solve(os.path.join(data_dir, lp_file))
    self.assertEqual(1.1084706, results.problem.lower_bound)
    self.assertEqual(1.35481947, results.problem.upper_bound)
    self.assertEqual(SolverStatus.aborted, results.solver.status)
    self.assertEqual(0.1, results.solver.system_time)
    self.assertEqual(0.11, results.solver.wallclock_time)
    self.assertEqual(TerminationCondition.maxTimeLimit, results.solver.termination_condition)
    self.assertEqual('Optimization terminated because the time expended exceeded the value specified in the seconds parameter.', results.solver.termination_message)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_bounded_subproblems, 0)
    self.assertEqual(results.solver.statistics.branch_and_bound.number_of_created_subproblems, 0)
    self.assertEqual(results.solver.statistics.black_box.number_of_iterations, 82)