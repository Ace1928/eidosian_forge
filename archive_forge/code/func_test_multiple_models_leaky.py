import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def test_multiple_models_leaky(self):
    with SolverFactory('gurobi_direct', manage_env=True) as opt:
        opt.solve(self.model)
        tmp = opt._solver_model
        opt.solve(self.model)
    with gp.Env():
        pass