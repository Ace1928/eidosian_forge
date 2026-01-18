import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_cuid_from_slice_3(self):
    """
        "3-level" slices. These test the ability of
        the slice-processing logic to handle multiple
        `get_item` calls in a hierarchy.
        """
    m = self._slice_model()
    _slice = m.b[:].b3[:, 'a', :].v2[1, :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b[*].b3[*,a,*].v2[1,*]')
    self.assertEqual(cuid, cuid_str)
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b[:].b3[:, 'a', :].v2
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b[*].b3[*,a,*].v2')
    self.assertEqual(cuid, cuid_str)
    self.assertListSameComponents(m, cuid, cuid_str)