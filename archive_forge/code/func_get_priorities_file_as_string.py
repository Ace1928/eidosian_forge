import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def get_priorities_file_as_string(self, mock_cplex_shell):
    with open(mock_cplex_shell._priorities_file_name, 'r') as ord_file:
        priorities_file = ord_file.read()
    return priorities_file