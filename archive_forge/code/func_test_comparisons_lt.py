import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_comparisons_lt(self):
    a = ComponentUID('foo.x[*]')
    a1 = ComponentUID('foo.x[1]')
    a2 = ComponentUID('foo.x[2]')
    aa = ComponentUID("foo.x['a']")
    a11 = ComponentUID('foo.x[1,1]')
    ae = ComponentUID('foo.x[**]')
    self.assertTrue(a < ae)
    self.assertTrue(a1 < ae)
    self.assertTrue(a1 < a)
    self.assertTrue(a1 < a2)
    self.assertTrue(a1 < aa)
    self.assertTrue(a1 < a11)
    self.assertTrue(a11 < a2)
    self.assertFalse(ae < a)
    self.assertFalse(ae < a1)
    self.assertFalse(a < a1)
    self.assertFalse(a2 < a1)
    self.assertFalse(aa < a1)
    self.assertFalse(a11 < a1)
    self.assertFalse(a2 < a11)
    x = ComponentUID('foo.x')
    xy = ComponentUID('foo.x.y')
    self.assertTrue(x < xy)
    self.assertFalse(xy < x)
    with self.assertRaisesRegex(TypeError, "'<' not supported between instances of 'ComponentUID' and 'int'"):
        a < 5