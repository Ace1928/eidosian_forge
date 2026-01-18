import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_delitem_component(self):
    init_all = list(self.m.bb[:, :, :])
    del self.m.bb[:, :, :]
    new_all = list(self.m.bb[:, :, :])
    self.assertEqual(len(init_all), 3 * 3 * 3)
    self.assertEqual(len(new_all), 0)
    init_all = list(self.m.b[:, :])
    init_tgt = list(self.m.b[1, :])
    del self.m.b[1, :]
    new_all = list(self.m.b[:, :])
    new_tgt = list(self.m.b[1, :])
    self.assertEqual(len(init_tgt), 3)
    self.assertEqual(len(init_all), 3 * 3)
    self.assertEqual(len(new_tgt), 0)
    self.assertEqual(len(new_all), 2 * 3)