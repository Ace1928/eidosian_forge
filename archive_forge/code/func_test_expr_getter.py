import pickle
import pyomo.common.unittest as unittest
from pyomo.core.expr import inequality, RangedExpression, EqualityExpression
from pyomo.kernel import pprint
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_tuple_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject
from pyomo.core.kernel.constraint import (
from pyomo.core.kernel.variable import variable
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression, data_expression
from pyomo.core.kernel.block import block
def test_expr_getter(self):
    c = constraint()
    self.assertIs(c.expr, None)
    v = variable()
    c.expr = 0 <= v
    self.assertIsNot(c.expr, None)
    self.assertEqual(c.lb, 0)
    self.assertIs(c.body, v)
    self.assertIs(c.ub, None)
    self.assertEqual(c.equality, False)
    c.expr = v <= 1
    self.assertIsNot(c.expr, None)
    self.assertIs(c.lb, None)
    self.assertIs(c.body, v)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.equality, False)
    c.expr = (0, v, 1)
    self.assertIsNot(c.expr, None)
    self.assertEqual(c.lb, 0)
    self.assertIs(c.body, v)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.equality, False)
    c.expr = v == 1
    self.assertIsNot(c.expr, None)
    self.assertEqual(c.lb, 1)
    self.assertIs(c.body, v)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.equality, True)
    c.expr = None
    self.assertIs(c.expr, None)
    self.assertIs(c.lb, None)
    self.assertIs(c.body, None)
    self.assertIs(c.ub, None)
    self.assertEqual(c.equality, False)