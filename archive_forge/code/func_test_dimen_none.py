import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def test_dimen_none(self):
    m = self.model()
    index_set = m.b0dn.index_set()
    index = ('c', 1, 10)
    pred_map = {0: m.d_none, 1: m.d_none, 2: m.d_none}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)
    index_set = m.b1dn.index_set()
    index = (1, 'c', 1, 10)
    pred_map = {0: m.time, 1: m.d_none, 2: m.d_none, 3: m.d_none}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)
    index_set = m.b1dnd2.index_set()
    index = (1, 'c', 1, 10, 'a', 1)
    pred_map = {0: m.time, 1: m.d_none, 2: m.d_none, 3: m.d_none, 4: m.d_2, 5: m.d_2}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)
    index_set = m.b2dn.index_set()
    index = (1, 0, 'd', 3)
    pred_map = {0: m.time, 1: m.space, 2: m.d_none, 3: m.d_none}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)
    index_set = m.b2dnd2.index_set()
    index = (1, 'c', 1, 10, 'b', 2, 0)
    pred_map = {0: m.time, 1: m.d_none, 2: m.d_none, 3: m.d_none, 4: m.d_2, 5: m.d_2, 6: m.space}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)
    index_set = m.dnd2b1.index_set()
    index = ('d', 3, 'b', 2, 1)
    pred_map = {0: m.d_none, 1: m.d_none, 2: m.d_2, 3: m.d_2, 4: m.time}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)
    index_set = m.b3dn.index_set()
    index = (1, 'a', 1, 'd', 3, 0, 'b', 2)
    pred_map = {0: m.time, 1: m.d_2, 2: m.d_2, 3: m.d_none, 4: m.d_none, 5: m.space, 6: m.d_2, 7: m.d_2}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)
    index_set = m.dn2.index_set()
    index = (1, 'c', 1, 10, 'b', 2, 'd', 3, 'a', 1)
    with self.assertRaises(RuntimeError) as cm:
        location_set_map = get_location_set_map(index, index_set)
    self.assertIn('multiple sets of dimen==None', str(cm.exception))