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
def test_nested_reference(self):
    m = ConcreteModel()
    m.I = Set(initialize=[1, 2])
    m.J = Set(initialize=[3, 4])

    @m.Block(m.I)
    def b(b, i):
        b.x = Var(b.model().J, bounds=(i, None))
    m.r = Reference(m.b[:].x[:])
    self.assertIs(m.r.ctype, Var)
    self.assertIsInstance(m.r.index_set(), SetProduct)
    self.assertIs(m.r.index_set().set_tuple[0], m.I)
    self.assertIs(m.r.index_set().set_tuple[1], m.J)
    self.assertEqual(len(m.r), 2 * 2)
    self.assertEqual(m.r[1, 3].lb, 1)
    self.assertEqual(m.r[2, 4].lb, 2)
    self.assertIn((1, 3), m.r)
    self.assertIn((2, 4), m.r)
    self.assertNotIn(0, m.r)
    self.assertNotIn((1, 0), m.r)
    self.assertNotIn((1, 3, 0), m.r)
    with self.assertRaises(KeyError):
        m.r[0]