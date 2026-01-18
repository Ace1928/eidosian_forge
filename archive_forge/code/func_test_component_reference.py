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
def test_component_reference(self):
    m = ConcreteModel()
    m.x = Var()
    m.r = Reference(m.x)
    self.assertIs(m.r.ctype, Var)
    self.assertIsNot(m.r.index_set(), m.x.index_set())
    self.assertIs(m.x.index_set(), UnindexedComponent_set)
    self.assertIs(m.r.index_set(), UnindexedComponent_ReferenceSet)
    self.assertEqual(len(m.r), 1)
    self.assertTrue(m.r.is_indexed())
    self.assertIn(None, m.r)
    self.assertNotIn(1, m.r)
    self.assertIs(m.r[None], m.x)
    with self.assertRaises(KeyError):
        m.r[1]
    m.s = Reference(m.x[:])
    self.assertIs(m.s.ctype, Var)
    self.assertIsNot(m.s.index_set(), m.x.index_set())
    self.assertIs(m.x.index_set(), UnindexedComponent_set)
    self.assertIs(type(m.s.index_set()), OrderedSetOf)
    self.assertEqual(len(m.s), 1)
    self.assertTrue(m.s.is_indexed())
    self.assertIn(None, m.s)
    self.assertNotIn(1, m.s)
    self.assertIs(m.s[None], m.x)
    with self.assertRaises(KeyError):
        m.s[1]
    m.y = Var([1, 2])
    m.t = Reference(m.y)
    self.assertIs(m.t.ctype, Var)
    self.assertIs(m.t.index_set(), m.y.index_set())
    self.assertEqual(len(m.t), 2)
    self.assertTrue(m.t.is_indexed())
    self.assertNotIn(None, m.t)
    self.assertIn(1, m.t)
    self.assertIn(2, m.t)
    self.assertIs(m.t[1], m.y[1])
    with self.assertRaises(KeyError):
        m.t[3]