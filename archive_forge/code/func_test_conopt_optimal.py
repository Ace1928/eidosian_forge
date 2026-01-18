import json
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import (
def test_conopt_optimal(self):
    with ReaderFactory('sol') as reader:
        if reader is None:
            raise IOError("Reader 'sol' is not registered")
        soln = reader(join(currdir, 'conopt_optimal.sol'))
        self.assertEqual(soln.solver.termination_condition, TerminationCondition.optimal)
        self.assertEqual(soln.solution.status, SolutionStatus.optimal)
        self.assertEqual(soln.solver.status, SolverStatus.ok)
        self.assertTrue(check_optimal_termination(soln))
        assert_optimal_termination(soln)