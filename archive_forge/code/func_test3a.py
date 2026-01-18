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
def test3a(self):
    m = ConcreteModel()
    m.x = Var(range(3), range(3), range(3), dense=True)
    names = set()
    for var in m.x[:, 1, :]:
        names.add(var.name)
    self.assertEqual(names, set(['x[0,1,0]', 'x[0,1,1]', 'x[0,1,2]', 'x[1,1,0]', 'x[1,1,1]', 'x[1,1,2]', 'x[2,1,0]', 'x[2,1,1]', 'x[2,1,2]']))