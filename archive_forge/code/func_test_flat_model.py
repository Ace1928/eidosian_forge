import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flat_model(self):
    m = ConcreteModel()
    m.T = ContinuousSet(bounds=(0, 1))
    m.x = Var()
    m.y = Var([1, 2])
    m.a = Var(m.T)
    m.b = Var(m.T, [1, 2])
    m.c = Var([3, 4], m.T)
    regular, time = flatten_dae_components(m, m.T, Var)
    regular_id = set((id(_) for _ in regular))
    self.assertEqual(len(regular), 3)
    self.assertIn(id(m.x), regular_id)
    self.assertIn(id(m.y[1]), regular_id)
    self.assertIn(id(m.y[2]), regular_id)
    ref_data = {self._hashRef(Reference(m.a[:])), self._hashRef(Reference(m.b[:, 1])), self._hashRef(Reference(m.b[:, 2])), self._hashRef(Reference(m.c[3, :])), self._hashRef(Reference(m.c[4, :]))}
    self.assertEqual(len(time), len(ref_data))
    for ref in time:
        self.assertIn(self._hashRef(ref), ref_data)