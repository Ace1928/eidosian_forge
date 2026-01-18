import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
def test_add_single_variable(self):
    """Test that the variable is added correctly to `solver_model`."""
    model = ConcreteModel()
    opt = SolverFactory('cplex', solver_io='python')
    opt._set_instance(model)
    self.assertEqual(opt._solver_model.variables.get_num(), 0)
    self.assertEqual(opt._solver_model.variables.get_num_binary(), 0)
    model.X = Var(within=Binary)
    var_interface = opt._solver_model.variables
    with unittest.mock.patch.object(var_interface, 'add', wraps=var_interface.add) as wrapped_add_call, unittest.mock.patch.object(var_interface, 'set_lower_bounds', wraps=var_interface.set_lower_bounds) as wrapped_lb_call, unittest.mock.patch.object(var_interface, 'set_upper_bounds', wraps=var_interface.set_upper_bounds) as wrapped_ub_call:
        opt._add_var(model.X)
        self.assertEqual(wrapped_add_call.call_count, 1)
        self.assertEqual(wrapped_add_call.call_args, ({'lb': [0], 'names': ['x1'], 'types': ['B'], 'ub': [1]},))
        self.assertFalse(wrapped_lb_call.called)
        self.assertFalse(wrapped_ub_call.called)
    self.assertEqual(opt._solver_model.variables.get_num(), 1)
    self.assertEqual(opt._solver_model.variables.get_num_binary(), 1)