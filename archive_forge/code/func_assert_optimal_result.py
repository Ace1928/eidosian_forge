import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
def assert_optimal_result(self, results):
    self.assertEqual(results.solver.status, SolverStatus.ok)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)