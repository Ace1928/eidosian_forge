import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def test_multiple_solvers(self):
    with SolverFactory('gurobi_direct') as opt1, SolverFactory('gurobi_direct') as opt2:
        opt1.solve(self.model)
        opt2.solve(self.model)