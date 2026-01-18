import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_invalid_string(self):
    m = self.make_model()
    m.p = Param(initialize='True', domain=Any)
    visitor = ExpressionBoundsVisitor()
    with self.assertRaisesRegex(ValueError, "'True' \\(str\\) is not a valid numeric type. Cannot compute bounds on expression."):
        lb, ub = visitor.walk_expression(m.p + m.y)