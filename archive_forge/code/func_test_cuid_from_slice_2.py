import pickle
from collections import namedtuple
from datetime import datetime
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.base.indexed_component import IndexedComponent
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.common.log import LoggingIntercept
def test_cuid_from_slice_2(self):
    """
        These are slices that describe a component
        at a "deeper level" than the original slice.
        """
    m = self._slice_model()
    _slice = m.b[:].b
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b[*].b')
    self.assertEqual(cuid, cuid_str)
    _slice = m.b[:].b1[:].v
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b[*].b1[*].v')
    self.assertEqual(cuid, cuid_str)
    _slice = m.b.b2[2, :].v
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[2,*].v')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[2, :].v1[:]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[2,*].v1[*]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[2, :].v1[1.1]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[2,*].v1[1.1]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertEqual(cuid, cuid_str)
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[2, :].vn[1, ..., :, 'b']
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[2,*].vn[1,**,*,b]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[2, :].vn[..., 'b']
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[2,*].vn[**,b]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[2, :].vn[..., ...]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[2,*].vn[**,**]')
    self.assertEqual(cuid, cuid_str)
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[2, :].vn[...]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[2,*].vn[**]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b2[...].v2[:, 'a']
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b2[**].v2[*,a]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b3[:, 'a', :].v1
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b3[*,a,*].v1')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b3[:, 'a', :].v2[1, 'a']
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b3[*,a,*].v2[1,a]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b3[:, 'a', :].v2[1, :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b3[*,a,*].v2[1,*]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.b3[:, 'a', :].vn[1, :, :, 'a', 1]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.b3[*,a,*].vn[1,*,*,a,1]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.bn['a', 'c', 3, :, :].vn[1, :, 3, 'a', :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.bn[a,c,3,*,*].vn[1,*,3,a,*]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.bn[...].vn[1, :, 3, 'a', :]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.bn[**].vn[1,*,3,a,*]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.bn[...].vn
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.bn[**].vn')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)
    _slice = m.b.bn[...].vn[...]
    cuid = ComponentUID(_slice)
    cuid_str = ComponentUID('b.bn[**].vn[**]')
    self.assertEqual(str(cuid), str(cuid_str))
    self.assertListSameComponents(m, cuid, cuid_str)