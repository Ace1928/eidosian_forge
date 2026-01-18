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
def test_bounds_getter_setter(self):
    c = constraint()
    self.assertEqual(c.bounds, (None, None))
    self.assertEqual(c.lb, None)
    self.assertEqual(c.ub, None)
    c.bounds = (1, 2)
    self.assertEqual(c.bounds, (1, 2))
    self.assertEqual(c.lb, 1)
    self.assertEqual(c.ub, 2)
    c.rhs = 3
    self.assertEqual(c.bounds, (3, 3))
    self.assertEqual(c.lb, 3)
    self.assertEqual(c.ub, 3)
    self.assertEqual(c.rhs, 3)
    with self.assertRaises(ValueError):
        c.bounds = (3, 3)
    self.assertEqual(c.bounds, (3, 3))
    self.assertEqual(c.lb, 3)
    self.assertEqual(c.ub, 3)
    self.assertEqual(c.rhs, 3)
    with self.assertRaises(ValueError):
        c.bounds = (2, 2)
    self.assertEqual(c.bounds, (3, 3))
    self.assertEqual(c.lb, 3)
    self.assertEqual(c.ub, 3)
    self.assertEqual(c.rhs, 3)
    with self.assertRaises(ValueError):
        c.bounds = (1, 2)
    self.assertEqual(c.bounds, (3, 3))
    self.assertEqual(c.lb, 3)
    self.assertEqual(c.ub, 3)
    self.assertEqual(c.rhs, 3)