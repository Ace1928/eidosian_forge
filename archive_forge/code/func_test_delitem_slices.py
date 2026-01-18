import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_delitem_slices(self):
    init_all = list(self.m.b[:, :].c[:, :].x[:])
    init_tgt = list(self.m.b[1, :].c[:, 4].x[:])
    del self.m.b[1, :].c[:, 4].x[:]
    new_all = list(self.m.b[:, :].c[:, :].x[:])
    new_tgt = list(self.m.b[1, :].c[:, 4].x[:])
    self.assertEqual(len(init_tgt), 3 * 3 * 3)
    self.assertEqual(len(init_all), 3 * 3 * (3 * 3) * 3)
    self.assertEqual(len(new_tgt), 0)
    self.assertEqual(len(new_all), 3 * 3 * (3 * 3) * 3 - 3 * 3 * 3)
    _slice = self.m.b[2, :].c[:, 4].x
    with self.assertRaisesRegex(KeyError, "Index 'bogus' is not valid for indexed component 'b\\[2,4\\]\\.c\\[1,4\\]\\.x'"):
        del _slice['bogus']
    _slice.key_errors_generate_exceptions = False
    del _slice['bogus']
    final_all = list(self.m.b[:, :].c[:, :].x[:])
    self.assertEqual(len(new_all), len(final_all))