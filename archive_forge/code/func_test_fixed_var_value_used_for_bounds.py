import math
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor, inf
from pyomo.contrib.fbbt.interval import _true, _false
from pyomo.core.expr import ExpressionBase, NumericExpression, BooleanExpression
def test_fixed_var_value_used_for_bounds(self):
    m = self.make_model()
    m.x.fix(3)
    visitor = ExpressionBoundsVisitor(use_fixed_var_values_as_bounds=True)
    lb, ub = visitor.walk_expression(m.x + m.y)
    self.assertEqual(lb, 6)
    self.assertEqual(ub, 8)
    self.assertEqual(len(visitor.leaf_bounds), 2)
    self.assertIn(m.x, visitor.leaf_bounds)
    self.assertIn(m.y, visitor.leaf_bounds)
    self.assertEqual(visitor.leaf_bounds[m.x], (3, 3))
    self.assertEqual(visitor.leaf_bounds[m.y], (3, 5))