import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.contrib.cp.repn.docplex_writer import docplex_available, LogicalToDoCplex
from pyomo.core.base.range import NumericRange
from pyomo.core.expr.numeric_expr import MinExpression, MaxExpression
from pyomo.core.expr.logical_expr import (
from pyomo.core.expr.relational_expr import NotEqualExpression
from pyomo.environ import (
def test_always_in(self):
    m = self.get_model()
    f = Pulse((m.i, 3)) + Step(m.i2[1].start_time, height=2) - Step(m.i2[2].end_time, height=-1) + Step(3, height=4)
    m.c = LogicalConstraint(expr=f.within((0, 3), (0, 10)))
    visitor = self.get_visitor()
    expr = visitor.walk_expression((m.c.expr, m.c, 0))
    self.assertIn(id(m.i), visitor.var_map)
    self.assertIn(id(m.i2[1]), visitor.var_map)
    self.assertIn(id(m.i2[2]), visitor.var_map)
    i = visitor.var_map[id(m.i)]
    i21 = visitor.var_map[id(m.i2[1])]
    i22 = visitor.var_map[id(m.i2[2])]
    self.assertTrue(expr[1].equals(cp.always_in(cp.pulse(i, 3) + cp.step_at_start(i21, 2) - cp.step_at_end(i22, -1) + cp.step_at(3, 4), interval=(0, 10), min=0, max=3)))