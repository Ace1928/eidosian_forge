import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_any_slice(self):
    m = ConcreteModel()
    m.x = Var(Any, dense=False)
    m.x[1] = 1
    m.x[1, 1] = 2
    m.x[2] = 3
    self.assertEqual(list((str(_) for _ in m.x[:])), ['x[1]', 'x[2]'])
    self.assertEqual(list((str(_) for _ in m.x[:, :])), ['x[1,1]'])
    self.assertEqual(list((str(_) for _ in m.x[...])), ['x[1]', 'x[1,1]', 'x[2]'])