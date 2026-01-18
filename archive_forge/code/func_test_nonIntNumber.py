import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_nonIntNumber(self):
    inf = float('inf')
    m = ConcreteModel()
    m.b = Block([inf, 'inf'])
    m.b[inf].x = x = Var()
    ref = 'b[inf].x'
    cuid = ComponentUID(x)
    self.assertEqual(cuid._cids, (('b', (inf,)), ('x', tuple())))
    self.assertTrue(cuid.matches(x))
    self.assertEqual(repr(ComponentUID(x)), ref)
    self.assertEqual(str(ComponentUID(x)), ref)
    cuid = ComponentUID(ref)
    self.assertEqual(cuid._cids, (('b', (inf,)), ('x', tuple())))
    self.assertTrue(cuid.matches(x))
    self.assertEqual(repr(ComponentUID(x)), ref)
    self.assertEqual(str(ComponentUID(x)), ref)
    ref = 'b:#inf.x'
    cuid = ComponentUID(ref)
    self.assertEqual(cuid._cids, (('b', (inf,)), ('x', tuple())))
    self.assertTrue(cuid.matches(x))
    self.assertEqual(ComponentUID(x).get_repr(1), ref)
    self.assertEqual(str(ComponentUID(x)), 'b[inf].x')
    m.b['inf'].x = x = Var()
    ref = "b['inf'].x"
    cuid = ComponentUID(x)
    self.assertEqual(cuid._cids, (('b', ('inf',)), ('x', tuple())))
    self.assertTrue(cuid.matches(x))
    self.assertEqual(repr(ComponentUID(x)), ref)
    self.assertEqual(str(ComponentUID(x)), ref)
    cuid = ComponentUID(ref)
    self.assertEqual(cuid._cids, (('b', ('inf',)), ('x', tuple())))
    self.assertTrue(cuid.matches(x))
    self.assertEqual(repr(ComponentUID(x)), ref)
    self.assertEqual(str(ComponentUID(x)), ref)
    ref = 'b:$inf.x'
    cuid = ComponentUID(ref)
    self.assertEqual(cuid._cids, (('b', ('inf',)), ('x', tuple())))
    self.assertTrue(cuid.matches(x))
    self.assertEqual(ComponentUID(x).get_repr(1), ref)
    self.assertEqual(str(ComponentUID(x)), "b['inf'].x")