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
def test_index_by_unhashable_type(self):
    m = ConcreteModel()
    m.x = Var([1, 2, 3], initialize=lambda m, x: 2 * x)
    self.assertRaisesRegex(TypeError, '.*', m.x.__getitem__, {})
    self.assertIs(m.x[[1]], m.x[1])
    m.y = Var([(1, 1), (1, 2)])
    self.assertIs(m.y[[1, 1]], m.y[1, 1])
    m.y[[1, 2]] = 5
    y12 = m.y[[1, 2]]
    self.assertEqual(y12.value, 5)
    m.y[[1, 2]] = 15
    self.assertIs(y12, m.y[[1, 2]])
    self.assertEqual(y12.value, 15)
    with self.assertRaisesRegex(KeyError, "Index '\\(2, 2\\)' is not valid for indexed component 'y'"):
        m.y[[2, 2]] = 5