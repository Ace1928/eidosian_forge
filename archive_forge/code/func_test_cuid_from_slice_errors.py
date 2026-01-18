import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_cuid_from_slice_errors(self):
    m = self._slice_model()
    m.b.comp = Reference(m.b.b1[:].v1)
    _slice = m.b[:].comp[1][1.1]
    with self.assertRaisesRegex(ValueError, '.*Two `get_item` calls.*'):
        cuid = ComponentUID(_slice)
    _slice = IndexedComponent_slice(m.b[:].component('v'), (IndexedComponent_slice.del_attribute, ('foo',)))
    with self.assertRaisesRegex(ValueError, "Cannot create a CUID from a slice that contains `set` or `del` calls: got call %s with argument \\('foo',\\)" % (IndexedComponent_slice.del_attribute,)):
        cuid = ComponentUID(_slice)