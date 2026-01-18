import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def test_close_global(self):
    opt1 = SolverFactory('gurobi_direct')
    opt2 = SolverFactory('gurobi_direct')
    try:
        opt1.solve(self.model)
        opt2.solve(self.model)
    finally:
        opt1.close()
        opt2.close_global()
    with gp.Env():
        pass