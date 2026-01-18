import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_explicit_skip_definition(self):
    model = ConcreteModel()
    model.idx = Set(initialize=[1, 2, 3])
    model.E = Expression(model.idx, rule=lambda m, i: Expression.Skip)
    self.assertEqual(len(model.E), 0)
    with self.assertRaises(KeyError):
        expr = model.E[1]
    model.E[1] = None
    expr = model.E[1]
    self.assertIs(type(expr), _GeneralExpressionData)
    self.assertIs(expr.expr, None)
    model.E[1] = 5
    self.assertIs(expr, model.E[1])
    self.assertEqual(model.E.extract_values(), {1: 5})
    model.E[2] = 6
    self.assertIsNot(expr, model.E[2])
    self.assertEqual(model.E.extract_values(), {1: 5, 2: 6})