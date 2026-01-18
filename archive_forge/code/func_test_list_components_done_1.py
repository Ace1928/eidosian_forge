import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_list_components_done_1(self):
    cuid = ComponentUID('b:*,*,*.c.a:*')
    ref = []
    cList = [str(ComponentUID(x)) for x in cuid.list_components(self.m)]
    self.assertEqual(sorted(cList), sorted(ref))