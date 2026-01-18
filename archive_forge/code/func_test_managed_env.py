import gc
from unittest.mock import patch
import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.environ import SolverFactory, ConcreteModel
from pyomo.opt import SolverStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.gurobi_direct import GurobiDirect
@unittest.skipIf(single_use_license(), reason='test requires multi-use license')
def test_managed_env(self):
    gp.setParam('IterationLimit', 100)
    with gp.Env(params={'IterationLimit': 0, 'Presolve': 0}) as use_env, patch('gurobipy.Env', return_value=use_env):
        with SolverFactory('gurobi_direct', manage_env=True) as opt:
            results = opt.solve(self.model)
            self.assertEqual(results.solver.status, SolverStatus.aborted)
            self.assertEqual(results.solver.termination_condition, TerminationCondition.maxIterations)