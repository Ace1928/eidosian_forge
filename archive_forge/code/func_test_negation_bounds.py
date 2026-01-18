import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_negation_bounds(self):
    m = self.make_model()
    visitor = ExpressionBoundsVisitor()
    lb, ub = visitor.walk_expression(-(m.y + 3 * m.x))
    self.assertEqual(lb, -17)
    self.assertEqual(ub, 3)