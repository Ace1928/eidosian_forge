import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_find_component_deprecated(self):
    ref = self.m.b[1, '2'].c.a[3]
    cuid = ComponentUID(ref)
    DEP_OUT = StringIO()
    with LoggingIntercept(DEP_OUT, 'pyomo.core'):
        self.assertTrue(cuid.find_component(self.m) is ref)
    self.assertIn('ComponentUID.find_component() is deprecated.', DEP_OUT.getvalue())