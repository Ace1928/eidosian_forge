import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_1level_model(self):
    m = ConcreteModel()
    m.T = ContinuousSet(bounds=(0, 1))

    @m.Block([1, 2], m.T)
    def B(b, i, t):
        b.x = Var(list(range(2 * i, 2 * i + 2)))
    regular, time = flatten_dae_components(m, m.T, Var)
    self.assertEqual(len(regular), 0)
    ref_data = {self._hashRef(Reference(m.B[1, :].x[2])), self._hashRef(Reference(m.B[1, :].x[3])), self._hashRef(Reference(m.B[2, :].x[4])), self._hashRef(Reference(m.B[2, :].x[5]))}
    self.assertEqual(len(time), len(ref_data))
    for ref in time:
        self.assertIn(self._hashRef(ref), ref_data)