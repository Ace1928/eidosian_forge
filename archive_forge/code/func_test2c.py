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
def test2c(self):
    m = ConcreteModel()
    m.x = Var(range(3), range(3), dense=True)
    names = set()
    for var in m.x[3, :]:
        names.add(var.name)
    self.assertEqual(names, set())