import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_nested_template_operation(self):
    m = self.m
    t = IndexTemplate(m.I)
    e = m.x[t + m.P[t + 1]]
    self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.x)
    self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(0), t)
    self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
    self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
    self.assertEqual(str(e), 'x[{I} + P[{I} + 1]]')