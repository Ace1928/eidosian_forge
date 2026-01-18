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
def test_expr_invalid_double_sided_inequality(self):
    x = variable()
    y = variable()
    c = constraint()
    c.expr = (0, x - y, 1)
    self.assertEqual(c.lb, 0)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.equality, False)
    with self.assertRaises(ValueError):
        c.expr = (x, x - y, 1)
    self.assertEqual(c.lb, 0)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.equality, False)
    with self.assertRaises(ValueError):
        c.expr = (0, x - y, y)
    self.assertEqual(c.lb, 0)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.equality, False)
    with self.assertRaises(ValueError):
        c.expr = (1, x - y, x)
    self.assertEqual(c.lb, 0)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.equality, False)
    with self.assertRaises(ValueError):
        c.expr = (y, x - y, 0)