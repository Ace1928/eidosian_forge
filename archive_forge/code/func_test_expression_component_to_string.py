import os
import platform
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.visitor import (
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual
def test_expression_component_to_string(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    m.e = Expression(expr=m.x * m.y)
    m.f = Expression(expr=m.e)
    e = m.x + m.f * m.y
    self.assertEqual('x + ((x*y))*y', str(e))
    self.assertEqual('x + ((x*y))*y', expression_to_string(e))