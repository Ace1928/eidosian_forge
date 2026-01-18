import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def test_init_default_env(self):
    with patch('gurobipy.Model') as PatchModel:
        with SolverFactory('gurobi_direct') as opt:
            opt.available()
            opt.available()
            PatchModel.assert_called_once_with()