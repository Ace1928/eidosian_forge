import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.environ import (
from pyomo.common.collections import ComponentSet
from pyomo.common.log import LoggingIntercept
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set import (
from pyomo.core.base.indexed_component import UnindexedComponent_set, IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.reference import (
def test_nested_scalars(self):
    m = ConcreteModel()
    m.b = Block()
    m.b.x = Var()
    m.r = Reference(m.b[:].x[:])
    self.assertEqual(len(m.r), 1)
    self.assertEqual(m.r.index_set().dimen, 2)
    base_sets = list(m.r.index_set().subsets())
    self.assertEqual(len(base_sets), 2)
    self.assertIs(type(base_sets[0]), OrderedSetOf)
    self.assertIs(type(base_sets[1]), OrderedSetOf)