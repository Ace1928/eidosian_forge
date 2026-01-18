import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_escapeChars(self):
    ref = "b['a\\n.b\\\\'].x"
    cuid = ComponentUID(ref)
    self.assertEqual(cuid._cids, (('b', ('a\n.b\\',)), ('x', tuple())))
    m = ConcreteModel()
    m.b = Block(['a\n.b\\'])
    m.b['a\n.b\\'].x = x = Var()
    self.assertTrue(cuid.matches(x))
    self.assertEqual(repr(ComponentUID(x)), ref)
    self.assertEqual(str(ComponentUID(x)), "b['a\\n.b\\\\'].x")