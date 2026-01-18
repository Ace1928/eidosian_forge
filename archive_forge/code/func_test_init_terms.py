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
def test_init_terms(self):
    v = variable(value=3)
    c = linear_constraint([], [], rhs=1)
    c.terms = ((v, 2),)
    self.assertEqual(c.lb, 1)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.rhs, 1)
    self.assertEqual(c.body(), 6)
    self.assertEqual(c(), 6)
    c = linear_constraint(terms=[(v, 2)], rhs=1)
    self.assertEqual(c.lb, 1)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.rhs, 1)
    self.assertEqual(c.body(), 6)
    self.assertEqual(c(), 6)
    terms = [(v, 2)]
    c = linear_constraint(terms=iter(terms), rhs=1)
    self.assertEqual(c.lb, 1)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.rhs, 1)
    self.assertEqual(c.body(), 6)
    self.assertEqual(c(), 6)
    c.terms = ()
    self.assertEqual(c.lb, 1)
    self.assertEqual(c.ub, 1)
    self.assertEqual(c.rhs, 1)
    self.assertEqual(c.body, 0)
    self.assertEqual(c(), 0)
    self.assertEqual(tuple(c.terms), ())