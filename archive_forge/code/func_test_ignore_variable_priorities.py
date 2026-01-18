import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def test_ignore_variable_priorities(self):
    model = self.get_mock_model_with_priorities()
    with SolverFactory('_mock_cplex') as opt:
        opt._presolve(model, priorities=False, keepfiles=True)
        self.assertIsNone(opt._priorities_file_name)
        self.assertNotIn('.ord', opt._command.script)