import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_list_components_scalar(self):
    cuid = ComponentUID('b:1,$2.c.a:3')
    ref = [str(ComponentUID(self.m.b[1, '2'].c.a[3]))]
    cList = [str(ComponentUID(x)) for x in cuid.list_components(self.m)]
    self.assertEqual(sorted(cList), sorted(ref))