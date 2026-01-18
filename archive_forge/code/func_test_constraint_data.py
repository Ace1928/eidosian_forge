import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
def test_constraint_data(self):
    solver_model = cplex.Cplex()
    solver_model.variables.add(lb=[-cplex.infinity, -cplex.infinity, -cplex.infinity], ub=[cplex.infinity, cplex.infinity, cplex.infinity], types=[solver_model.variables.type.continuous, solver_model.variables.type.continuous, solver_model.variables.type.continuous], names=['var1', 'var2', 'var3'])
    con_data = _LinearConstraintData(solver_model)
    con_data.add(cplex_expr=_CplexExpr(variables=[0, 1], coefficients=[10, 100]), sense='L', rhs=0, range_values=0, name='c1')
    con_data.add(cplex_expr=_CplexExpr(variables=[0], coefficients=[-30]), sense='G', rhs=1, range_values=0, name='c2')
    con_data.add(cplex_expr=_CplexExpr(variables=[1], coefficients=[80]), sense='E', rhs=2, range_values=0, name='c3')
    con_data.add(cplex_expr=_CplexExpr(variables=[2], coefficients=[50]), sense='R', rhs=3, range_values=10, name='c4')
    self.assertEqual(solver_model.linear_constraints.get_num(), 0)
    con_data.store_in_cplex()
    self.assertEqual(solver_model.linear_constraints.get_num(), 4)