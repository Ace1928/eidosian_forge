import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_end_after_start(self):
    m = self.get_model()
    m.c = LogicalConstraint(expr=m.a.end_time.after(m.b.start_time, delay=-2))
    self.assertIsInstance(m.c.expr, BeforeExpression)
    self.assertEqual(len(m.c.expr.args), 3)
    self.assertIs(m.c.expr.args[0], m.b.start_time)
    self.assertIs(m.c.expr.args[1], m.a.end_time)
    self.assertEqual(m.c.expr.delay, -2)
    self.assertEqual(str(m.c.expr), 'b.start_time - 2 <= a.end_time')