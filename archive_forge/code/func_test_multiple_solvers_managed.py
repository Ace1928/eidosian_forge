import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
@unittest.skipIf(single_use_license(), reason='test requires multi-use license')
def test_multiple_solvers_managed(self):
    with SolverFactory('gurobi_direct', manage_env=True) as opt1, SolverFactory('gurobi_direct', manage_env=True) as opt2:
        results1 = opt1.solve(self.model)
        self.assert_optimal_result(results1)
        results2 = opt2.solve(self.model)
        self.assert_optimal_result(results2)