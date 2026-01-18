import copy
from io import StringIO
from pyomo.core.expr import expr_common
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.environ import (
from pyomo.core.base.expression import _GeneralExpressionData
from pyomo.core.expr.compare import compare_expressions, assertExpressionsEqual
from pyomo.common.tee import capture_output
def test_indexed_construct_rule(self):
    model = ConcreteModel()
    model.Index = Set(initialize=[1, 2, 3])

    def _some_rule(model, i):
        if i == 1:
            return Expression.Skip
        else:
            return i
    model.E = Expression(model.Index, rule=_some_rule)
    self.assertEqual(model.E.extract_values(), {2: 2, 3: 3})
    self.assertEqual(len(model.E), 2)