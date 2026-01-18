import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_hash_eqality(self):
    m = self.m
    a = m.b[1, :].c[:, ..., 4].x
    b = m.b[1, :].c[1, ..., :].x
    self.assertNotEqual(a, b)
    self.assertNotEqual(a, m)
    self.assertEqual(a, a)
    self.assertEqual(a, m.b[1, :].c[:, ..., 4].x)
    _set = set([a, b])
    self.assertEqual(len(_set), 2)
    _set.add(m.b[1, :].c[:, ..., 4].x)
    self.assertEqual(len(_set), 2)
    _set.add(m.b[1, :].c[:, 4].x)
    self.assertEqual(len(_set), 3)