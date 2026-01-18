import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_invalid_slices(self):
    m = self.m
    m.x = Var()
    for var in m.x[:]:
        self.assertIs(var, m.x)
    with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
        _slicer = m.b[:]
    with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
        _slicer = m.b[:, :, :]
    with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
        _slicer = m.b[:, :, :, ...]
    _slicer = m.b[:, :, ...].c[:, :, :].x
    with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
        list(_slicer)
    _slicer = m.b[2, :].c[:].x
    with self.assertRaisesRegex(IndexError, 'Index .* contains an invalid number of entries for component .*'):
        list(_slicer)