import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
def test_add_block_containing_single_variable(self):
    """Test that the variable is added correctly to `solver_model`."""
    model = ConcreteModel()
    opt = SolverFactory('cplex', solver_io='python')
    opt._set_instance(model)
    self.assertEqual(opt._solver_model.variables.get_num(), 0)
    self.assertEqual(opt._solver_model.variables.get_num_binary(), 0)
    model.X = Var(within=Binary)
    with unittest.mock.patch.object(opt._solver_model.variables, 'add', wraps=opt._solver_model.variables.add) as wrapped_add_call:
        opt._add_block(model)
        self.assertEqual(wrapped_add_call.call_count, 1)
        self.assertEqual(wrapped_add_call.call_args, ({'lb': [0], 'names': ['x1'], 'types': ['B'], 'ub': [1]},))
    self.assertEqual(opt._solver_model.variables.get_num(), 1)
    self.assertEqual(opt._solver_model.variables.get_num_binary(), 1)