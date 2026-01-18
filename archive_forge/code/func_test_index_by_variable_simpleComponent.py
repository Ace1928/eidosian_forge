import os
from os.path import abspath, dirname
from pyomo.common import DeveloperError
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.environ import ConcreteModel, Var, Param, Set, value, Integers
from pyomo.core.base.set import FiniteSetOf, OrderedSetOf
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.expr import GetItemExpression
from pyomo.core import SortComponents
def test_index_by_variable_simpleComponent(self):
    m = ConcreteModel()
    m.i = Var(initialize=2, domain=Integers)
    m.x = Var([1, 2, 3], initialize=lambda m, x: 2 * x)
    self.assertEqual(value(m.x[2]), 4)
    thing = m.x[m.i]
    self.assertIsInstance(thing, GetItemExpression)
    self.assertEqual(len(thing.args), 2)
    self.assertIs(thing.args[0], m.x)
    self.assertIs(thing.args[1], m.i)
    idx_expr = 2 * m.i + 1
    thing = m.x[idx_expr]
    self.assertIsInstance(thing, GetItemExpression)
    self.assertEqual(len(thing.args), 2)
    self.assertIs(thing.args[0], m.x)
    self.assertIs(thing.args[1], idx_expr)