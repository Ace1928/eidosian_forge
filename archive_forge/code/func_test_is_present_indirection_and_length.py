import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_is_present_indirection_and_length(self):
    m = self.get_model()
    m.y = Var(domain=Integers, bounds=[1, 2])
    m.c = LogicalConstraint(expr=m.i2[m.y].is_present.land(m.i2[m.y].length >= 7))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.expr, m.c, 0))
    self.assertIn(id(m.y), visitor.var_map)
    self.assertIn(id(m.i2[1]), visitor.var_map)
    self.assertIn(id(m.i2[2]), visitor.var_map)
    y = visitor.var_map[id(m.y)]
    i21 = visitor.var_map[id(m.i2[1])]
    i22 = visitor.var_map[id(m.i2[2])]
    self.assertTrue(expr[1].equals(cp.logical_and(cp.element([cp.presence_of(i21), cp.presence_of(i22)], 0 + 1 * (y - 1) // 1) == True, cp.less_or_equal(7, cp.element([cp.length_of(i21), cp.length_of(i22)], 0 + 1 * (y - 1) // 1)))))