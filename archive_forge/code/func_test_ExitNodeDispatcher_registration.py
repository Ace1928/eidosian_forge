import logging
import math
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
from pyomo.environ import (
import pyomo.repn.util
from pyomo.repn.util import (
def test_ExitNodeDispatcher_registration(self):
    end = ExitNodeDispatcher({ProductExpression: lambda v, n, d1, d2: d1 * d2, Expression: lambda v, n, d: d})
    self.assertEqual(len(end), 2)
    node = ProductExpression((3, 4))
    self.assertEqual(end[node.__class__](None, node, *node.args), 12)
    self.assertEqual(len(end), 2)
    node = Expression(initialize=5)
    node.construct()
    self.assertEqual(end[node.__class__](None, node, *node.args), 5)
    self.assertEqual(len(end), 3)
    self.assertIn(node.__class__, end)
    node = NPV_ProductExpression((6, 7))
    self.assertEqual(end[node.__class__](None, node, *node.args), 42)
    self.assertEqual(len(end), 4)
    self.assertIn(NPV_ProductExpression, end)
    end[SumExpression, 2] = lambda v, n, *d: 2 * sum(d)
    self.assertEqual(len(end), 5)
    node = SumExpression((1, 2, 3))
    self.assertEqual(end[node.__class__, 2](None, node, *node.args), 12)
    self.assertEqual(len(end), 5)
    with self.assertRaisesRegex(DeveloperError, "(?s)Base expression key '\\(<class.*'pyomo.core.expr.numeric_expr.SumExpression'>, 3\\)' not found when.*inserting dispatcher for node 'SumExpression' while walking.*expression tree."):
        end[node.__class__, 3](None, node, *node.args)
    self.assertEqual(len(end), 5)
    end[SumExpression] = lambda v, n, *d: sum(d)
    self.assertEqual(len(end), 6)
    self.assertIn(SumExpression, end)
    self.assertEqual(end[node.__class__, 1](None, node, *node.args), 6)
    self.assertEqual(len(end), 7)
    self.assertIn((SumExpression, 1), end)
    self.assertEqual(end[node.__class__, 3, 4, 5, 6](None, node, *node.args), 6)
    self.assertEqual(len(end), 7)
    self.assertNotIn((SumExpression, 3, 4, 5, 6), end)

    class NewProductExpression(ProductExpression):
        pass
    node = NewProductExpression((6, 7))
    self.assertEqual(end[node.__class__](None, node, *node.args), 42)
    self.assertEqual(len(end), 8)
    self.assertIn(NewProductExpression, end)

    class UnknownExpression(NumericExpression):
        pass
    node = UnknownExpression((6, 7))
    with self.assertRaisesRegex(DeveloperError, ".*Unexpected expression node type 'UnknownExpression'"):
        end[node.__class__](None, node, *node.args)
    self.assertEqual(len(end), 9)
    self.assertIn(UnknownExpression, end)
    node = UnknownExpression((6, 7))
    with self.assertRaisesRegex(DeveloperError, ".*Unexpected expression node type 'UnknownExpression'"):
        end[node.__class__, 6, 7](None, node, *node.args)
    self.assertEqual(len(end), 10)
    self.assertIn((UnknownExpression, 6, 7), end)