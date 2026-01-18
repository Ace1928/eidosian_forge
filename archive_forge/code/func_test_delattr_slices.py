import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_delattr_slices(self):
    self.m.b[1, :].c[:, 4].x.foo = 10
    self.assertEqual(len(list(self.m.b[1, :].c[:, 4].x)), 3 * 3)
    self.assertEqual(sum(list(self.m.b[1, :].c[:, 4].x.foo)), 10 * 3 * 3)
    self.assertEqual(sum(list((1 if hasattr(x, 'foo') else 0 for x in self.m.b[:, :].c[:, :].x))), 3 * 3)
    _slice = self.m.b[1, :].c[:, 4].x.foo
    _slice._call_stack[-1] = (IndexedComponent_slice.del_attribute, _slice._call_stack[-1][1])
    list(_slice)
    self.assertEqual(sum(list((1 if hasattr(x, 'foo') else 0 for x in self.m.b[:, :].c[:, :].x))), 0)
    with self.assertRaisesRegex(AttributeError, 'foo'):
        list(_slice)
    _slice.attribute_errors_generate_exceptions = False
    list(_slice)