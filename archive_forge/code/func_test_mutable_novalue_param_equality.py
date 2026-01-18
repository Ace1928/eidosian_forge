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
def test_mutable_novalue_param_equality(self):
    x = variable()
    p = parameter()
    p.value = None
    c = constraint(expr=x - p == 0)
    self.assertEqual(c.equality, True)
    c = constraint(expr=x == p)
    self.assertIs(c.upper, p)
    self.assertEqual(c.equality, True)
    c = constraint(expr=x + 1 == p)
    self.assertEqual(c.equality, True)
    c = constraint(expr=x + 1 == (p + 1) ** 2)
    self.assertEqual(c.equality, True)
    c = constraint(expr=x == p + 1)
    self.assertEqual(c.equality, True)
    c = constraint(expr=(x, p))
    self.assertIs(c.upper, p)
    self.assertIs(c.lower, p)
    self.assertIs(c.rhs, p)
    self.assertIs(c.body, x)
    self.assertEqual(c.equality, True)
    c = constraint(expr=(p, x))
    self.assertIs(c.upper, p)
    self.assertIs(c.lower, p)
    self.assertIs(c.rhs, p)
    self.assertIs(c.body, x)
    self.assertEqual(c.equality, True)
    c = constraint(expr=EqualityExpression((p, x)))
    self.assertIs(c.upper, p)
    self.assertIs(c.lower, p)
    self.assertIs(c.rhs, p)
    self.assertIs(c.body, x)
    self.assertEqual(c.equality, True)
    c = constraint(expr=EqualityExpression((x, p)))
    self.assertIs(c.upper, p)
    self.assertIs(c.lower, p)
    self.assertIs(c.rhs, p)
    self.assertIs(c.body, x)
    self.assertEqual(c.equality, True)