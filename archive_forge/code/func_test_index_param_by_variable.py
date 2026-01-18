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
def test_index_param_by_variable(self):
    m = ConcreteModel()
    m.i = Var(initialize=2, domain=Integers)
    m.p = Param([1, 2, 3], initialize=lambda m, x: 2 * x)
    thing = m.p[m.i]
    self.assertIsInstance(thing, GetItemExpression)
    self.assertEqual(len(thing.args), 2)
    self.assertIs(thing.args[0], m.p)
    self.assertIs(thing.args[1], m.i)
    idx_expr = 2 ** m.i + 1
    thing = m.p[idx_expr]
    self.assertIsInstance(thing, GetItemExpression)
    self.assertEqual(len(thing.args), 2)
    self.assertIs(thing.args[0], m.p)
    self.assertIs(thing.args[1], idx_expr)