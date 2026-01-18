import pyomo.common.unittest as unittest
from pyomo.core import ConcreteModel, Var, Expression, Block, RangeSet, Any
import pyomo.core.expr as EXPR
from pyomo.core.base.expression import _ExpressionData
from pyomo.gdp.util import (
from pyomo.gdp import Disjunct, Disjunction
def test_clone_without_expression_components(self):
    m = ConcreteModel()
    m.x = Var(initialize=5)
    m.y = Var(initialize=3)
    m.e = Expression(expr=m.x ** 2 + m.x - 1)
    base = m.x ** 2 + 1
    test = clone_without_expression_components(base, {})
    self.assertIs(base, test)
    self.assertEqual(base(), test())
    test = clone_without_expression_components(base, {id(m.x): m.y})
    self.assertEqual(3 ** 2 + 1, test())
    base = m.e
    test = clone_without_expression_components(base, {})
    self.assertIsNot(base, test)
    self.assertEqual(base(), test())
    self.assertIsInstance(base, _ExpressionData)
    self.assertIsInstance(test, EXPR.SumExpression)
    test = clone_without_expression_components(base, {id(m.x): m.y})
    self.assertEqual(3 ** 2 + 3 - 1, test())
    base = m.e + m.x
    test = clone_without_expression_components(base, {})
    self.assertIsNot(base, test)
    self.assertEqual(base(), test())
    self.assertIsInstance(base, EXPR.SumExpression)
    self.assertIsInstance(test, EXPR.SumExpression)
    self.assertIsInstance(base.arg(0), _ExpressionData)
    self.assertIsInstance(test.arg(0), EXPR.SumExpression)
    test = clone_without_expression_components(base, {id(m.x): m.y})
    self.assertEqual(3 ** 2 + 3 - 1 + 3, test())