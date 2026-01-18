import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
def test_add_block_containing_multiple_constraints(self):
    model = ConcreteModel()
    model.X = Var(within=Binary)
    opt = SolverFactory('cplex', solver_io='python')
    opt._set_instance(model)
    self.assertEqual(opt._solver_model.linear_constraints.get_num(), 0)
    model.B = Block()
    model.B.C1 = Constraint(expr=model.X == 1)
    model.B.C2 = Constraint(expr=model.X <= 1)
    model.B.C3 = Constraint(expr=model.X >= 1)
    con_interface = opt._solver_model.linear_constraints
    with unittest.mock.patch.object(con_interface, 'add', wraps=con_interface.add) as wrapped_add_call:
        opt._add_block(model.B)
        self.assertEqual(wrapped_add_call.call_count, 1)
        self.assertEqual(wrapped_add_call.call_args, ({'lin_expr': [[[0], (1,)], [[0], (1,)], [[0], (1,)]], 'names': ['x2', 'x3', 'x4'], 'range_values': [0.0, 0.0, 0.0], 'rhs': [1.0, 1.0, 1.0], 'senses': ['E', 'L', 'G']},))
    self.assertEqual(opt._solver_model.linear_constraints.get_num(), 3)