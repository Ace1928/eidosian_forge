import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_interval_var_fixed_length(self):
    m = ConcreteModel()
    m.i = IntervalVar(start=(2, 7), end=(6, 11), optional=True)
    m.i.length.fix(4)
    m.silly = LogicalConstraint(expr=m.i.is_present)
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.silly.expr, m.silly, 0))
    self.assertIn(id(m.i), visitor.var_map)
    i = visitor.var_map[id(m.i)]
    self.assertTrue(i.is_optional())
    self.assertEqual(i.get_length(), (4, 4))
    self.assertEqual(i.get_start(), (2, 7))
    self.assertEqual(i.get_end(), (6, 11))