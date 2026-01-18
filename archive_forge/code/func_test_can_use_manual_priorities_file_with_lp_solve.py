import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def test_can_use_manual_priorities_file_with_lp_solve(self):
    """Test that we can pass an LP file (not a pyomo model) along with a priorities file to `.solve()`"""
    model = self.get_mock_model_with_priorities()
    with SolverFactory('_mock_cplex') as pre_opt:
        pre_opt._presolve(model, priorities=True, keepfiles=True)
        lp_file = pre_opt._problem_files[0]
        priorities_file_name = pre_opt._priorities_file_name
        with open(priorities_file_name, 'r') as ord_file:
            provided_priorities_file = ord_file.read()
    with SolverFactory('_mock_cplex') as opt:
        opt._presolve(lp_file, priorities=True, priorities_file=priorities_file_name, keepfiles=True)
        self.assertIn('.ord', opt._command.script)
        with open(opt._priorities_file_name, 'r') as ord_file:
            priorities_file = ord_file.read()
    self.assertEqual(priorities_file, provided_priorities_file)