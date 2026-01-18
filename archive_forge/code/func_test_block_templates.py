import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_block_templates(self):
    m = ConcreteModel()
    m.T = RangeSet(3)

    @m.Block(m.T)
    def b(b, i):
        b.x = Var(initialize=i)

        @b.Block(m.T)
        def bb(bb, j):
            bb.I = RangeSet(i * j)
            bb.y = Var(bb.I, initialize=lambda m, i: i)
    t = IndexTemplate(m.T)
    e = m.b[t].x
    self.assertIs(type(e), EXPR.Numeric_GetAttrExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(type(e.arg(0)), EXPR.NPV_Structural_GetItemExpression)
    self.assertIs(e.arg(0).arg(0), m.b)
    self.assertEqual(e.arg(0).nargs(), 2)
    self.assertIs(e.arg(0).arg(1), t)
    self.assertEqual(str(e), 'b[{T}].x')
    t.set_value(2)
    v = e()
    self.assertIn(type(v), (int, float))
    self.assertEqual(v, 2)
    self.assertIs(resolve_template(e), m.b[2].x)
    t.set_value()
    e = m.b[t].bb[t].y[1]
    self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertEqual(str(e), 'b[{T}].bb[{T}].y[1]')
    t.set_value(2)
    v = e()
    self.assertIn(type(v), (int, float))
    self.assertEqual(v, 1)
    self.assertIs(resolve_template(e), m.b[2].bb[2].y[1])