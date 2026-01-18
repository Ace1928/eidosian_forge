import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_expression_with_mutable_param(self):
    m = ConcreteModel()
    m.x = Var(domain=Integers, bounds=(2, 3))
    m.p = Param(initialize=4, mutable=True)
    e = m.p * m.x
    visitor = self.get_visitor()
    expr = visitor.walk_expression((e, e, 0))
    self.assertIn(id(m.x), visitor.var_map)
    x = visitor.var_map[id(m.x)]
    self.assertTrue(expr[1].equals(4 * x))