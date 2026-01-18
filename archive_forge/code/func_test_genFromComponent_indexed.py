import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_genFromComponent_indexed(self):
    cuid = ComponentUID(self.m.b[1, '2'].c.a)
    self.assertEqual(cuid._cids, (('b', (1, '2')), ('c', tuple()), ('a', ())))