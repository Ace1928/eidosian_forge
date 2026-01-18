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
def test_component_data_reference_clone(self):
    m = ConcreteModel()
    m.b = Block()
    m.b.x = Var([1, 2])
    m.c = Block()
    m.c.r1 = Reference(m.b.x[2])
    m.c.r2 = Reference(m.b.x)
    self.assertIs(m.c.r1[None], m.b.x[2])
    m.d = m.c.clone()
    self.assertIs(m.d.r1[None], m.b.x[2])
    self.assertIs(m.d.r2[1], m.b.x[1])
    self.assertIs(m.d.r2[2], m.b.x[2])
    i = m.clone()
    self.assertIs(i.c.r1[None], i.b.x[2])
    self.assertIs(i.c.r2[1], i.b.x[1])
    self.assertIs(i.c.r2[2], i.b.x[2])
    self.assertIsNot(i.c.r1[None], m.b.x[2])
    self.assertIsNot(i.c.r2[1], m.b.x[1])
    self.assertIsNot(i.c.r2[2], m.b.x[2])
    self.assertIs(i.d.r1[None], i.b.x[2])
    self.assertIs(i.d.r2[1], i.b.x[1])
    self.assertIs(i.d.r2[2], i.b.x[2])