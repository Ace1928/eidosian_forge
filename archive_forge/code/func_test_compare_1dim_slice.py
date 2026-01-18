import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_compare_1dim_slice(self):
    m = ConcreteModel()
    m.I = Set(initialize=range(2))
    m.J = Set(initialize=range(2, 4))
    m.K = Set(initialize=['a', 'b'])

    @m.Block(m.I, m.J)
    def b(b, i, j):
        b.v = Var(m.K)
    self.assertEqual(m.b[0, :].v[:], m.b[0, :].v[:])
    self.assertNotEqual(m.b[0, :].v[:], m.b[0, :].v['a'])