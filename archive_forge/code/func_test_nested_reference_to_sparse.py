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
def test_nested_reference_to_sparse(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1])

    @m.Block(m.I)
    def b(b, i):
        b.x = Var(b.model().I, dense=False)
    m.xx = Reference(m.b[:].x[:], ctype=Var)
    m.I.add(2)
    m.I.add(3)
    self.assertEqual(len(m.b), 1)
    self.assertEqual(len(m.b[1].x), 0)
    self.assertIn(1, m.b)
    self.assertNotIn((1, 1), m.xx)
    self.assertIn(1, m.b[1].x.index_set())
    self.assertIn((1, 1), m.xx.index_set())
    self.assertEqual(len(m.b), 1)
    self.assertEqual(len(m.b[1].x), 0)
    m.xx[1, 2]
    self.assertEqual(len(m.b), 1)
    self.assertEqual(len(m.b[1].x), 1)
    self.assertIs(m.xx[1, 2], m.b[1].x[2])
    self.assertEqual(len(m.b), 1)
    self.assertEqual(len(m.b[1].x), 1)
    m.xx[1, 3] = 5
    self.assertEqual(len(m.b), 1)
    self.assertEqual(len(m.b[1].x), 2)
    self.assertIs(m.xx[1, 3], m.b[1].x[3])
    self.assertEqual(value(m.b[1].x[3]), 5)
    self.assertEqual(len(m.b), 1)
    self.assertEqual(len(m.b[1].x), 2)
    m.xx.add((1, 1))
    self.assertEqual(len(m.b), 1)
    self.assertEqual(len(m.b[1].x), 3)
    self.assertIs(m.xx[1, 1], m.b[1].x[1])
    self.assertEqual(len(m.b), 1)
    self.assertEqual(len(m.b[1].x), 3)
    self.assertEqual(len(m.xx), 3)
    m.xx[2, 2] = 10
    self.assertEqual(len(m.b), 2)
    self.assertEqual(len(list(m.b[2].component_objects())), 1)
    self.assertEqual(len(m.xx), 4)
    self.assertIs(m.xx[2, 2], m.b[2].x[2])
    self.assertEqual(value(m.b[2].x[2]), 10)