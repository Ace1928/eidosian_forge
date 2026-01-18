import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def test_1d_replacement(self):
    m = self.model()
    index_set = m.b0.index_set()
    index = None
    location_set_map = get_location_set_map(index, index_set)
    sets = ComponentSet()
    pred_index = (None,)
    new_index = replace_indices(index, location_set_map, sets)
    self.assertEqual(pred_index, new_index)
    index_set = m.b1.index_set()
    index = 1
    location_set_map = get_location_set_map(index, index_set)
    sets = ComponentSet()
    pred_index = (1,)
    new_index = replace_indices(index, location_set_map, sets)
    self.assertEqual(pred_index, new_index)
    sets = ComponentSet((m.time,))
    pred_index = (slice(None),)
    new_index = replace_indices(index, location_set_map, sets)
    self.assertEqual(pred_index, new_index)
    index_set = m.b2.index_set()
    index = (1, 2)
    location_set_map = get_location_set_map(index, index_set)
    sets = ComponentSet()
    pred_index = (1, 2)
    new_index = replace_indices(index, location_set_map, sets)
    self.assertEqual(pred_index, new_index)
    sets = ComponentSet((m.space,))
    pred_index = (1, slice(None))
    new_index = replace_indices(index, location_set_map, sets)
    self.assertEqual(pred_index, new_index)
    sets = ComponentSet((m.space, m.time))
    pred_index = (slice(None), slice(None))
    new_index = replace_indices(index, location_set_map, sets)
    self.assertEqual(pred_index, new_index)