import os
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core import (
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import (
def get_mock_model(self):
    model = pmo.block()
    model.x = pmo.variable(domain=Binary)
    model.con = pmo.constraint(expr=model.x >= 1)
    model.obj = pmo.objective(expr=model.x)
    return model