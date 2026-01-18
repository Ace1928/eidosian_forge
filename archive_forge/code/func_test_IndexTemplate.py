import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_IndexTemplate(self):
    m = self.m
    i = IndexTemplate(m.I)
    with self.assertRaisesRegex(TemplateExpressionError, 'Evaluating uninitialized IndexTemplate'):
        value(i)
    self.assertEqual(str(i), '{I}')
    i.set_value(5)
    self.assertEqual(value(i), 5)
    self.assertIs(resolve_template(i), 5)