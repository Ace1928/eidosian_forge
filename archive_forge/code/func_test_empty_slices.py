import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_empty_slices(self):
    _slicer = self.m.b[1, :].c[:, 1].x
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, [])
    _slicer = self.m.b[1, :].c[:, 4].x[1]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    _slicer.key_errors_generate_exceptions = False
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, [])
    _slicer = self.m.b[1, :].c[:, 4].y
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    _slicer.attribute_errors_generate_exceptions = False
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, [])
    _slicer = self.m.b[1, :].c[:, 4].component('y', False)
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    _slicer.call_errors_generate_exceptions = False
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, [])
    _slicer = self.m.b[1, :].c[:, 4].x[1]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    _slicer.key_errors_generate_exceptions = True
    self.assertRaises(KeyError, _slicer.next)
    _slicer = self.m.b[1, :].c[:, 4].y
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    _slicer.attribute_errors_generate_exceptions = True
    self.assertRaises(AttributeError, _slicer.next)
    _slicer = self.m.b[1, :].c[:, 4].component('y', False)
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    _slicer.call_errors_generate_exceptions = True
    self.assertRaises(TypeError, _slicer.next)
    _slicer = self.m.b[1, :].c[:, 4].component()
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    _slicer.call_errors_generate_exceptions = True
    self.assertRaises(TypeError, _slicer.next)