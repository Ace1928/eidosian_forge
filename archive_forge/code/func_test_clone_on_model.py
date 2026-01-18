import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_clone_on_model(self):
    m = self.m
    m.slicer = m.b[1, :].c[:, 4].x
    n = m.clone()
    self.assertIsNot(m, n)
    self.assertIsNot(m.slicer, n.slicer)
    self.assertIsNot(m.slicer._call_stack, n.slicer._call_stack)
    self.assertIs(type(m.slicer._call_stack), type(n.slicer._call_stack))
    self.assertEqual(len(m.slicer._call_stack), len(n.slicer._call_stack))
    ref = ['b[1,4].c[1,4].x', 'b[1,4].c[2,4].x', 'b[1,4].c[3,4].x', 'b[1,5].c[1,4].x', 'b[1,5].c[2,4].x', 'b[1,5].c[3,4].x', 'b[1,6].c[1,4].x', 'b[1,6].c[2,4].x', 'b[1,6].c[3,4].x']
    self.assertEqual([str(x) for x in m.slicer], ref)
    self.assertEqual([str(x) for x in n.slicer], ref)
    for x, y in zip(iter(m.slicer), iter(n.slicer)):
        self.assertIs(type(x), type(y))
        self.assertEqual(x.name, y.name)
        self.assertIsNot(x, y)
        self.assertIs(x.model(), m)
        self.assertIs(y.model(), n)