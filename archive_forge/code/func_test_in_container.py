import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_in_container(self):
    a = ComponentUID('foo.bar[*]')
    b = ComponentUID('baz')
    c = ComponentUID('baz.bar')
    D = {a: 1, b: 2}
    self.assertTrue(a in D)
    self.assertTrue(b in D)
    self.assertFalse(c in D)
    self.assertTrue(ComponentUID('foo.bar[*]') in D)
    self.assertTrue(ComponentUID('baz') in D)