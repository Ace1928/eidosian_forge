import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_nonTemplates(self):
    m = self.m
    self.assertIs(resolve_template(m.x[1]), m.x[1])
    e = m.x[1] + m.x[2]
    self.assertIs(resolve_template(e), e)