import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flatten_m3_1d(self):
    m = self._model3_nd_sets_normalizeflatten()
    sets = ComponentSet((m.d1,))
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    assert len(sets_list) == len(comps_list)
    assert len(sets_list) == 3
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is m.d1:
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.v_12[:, i2])) for i2 in m.d2))
            ref_data.update((self._hashRef(Reference(m.v_212[i2a, :, i2b])) for i2a in m.d2 for i2b in m.d2))
            ref_data.update((self._hashRef(Reference(m.v_12n[:, i2, i_n])) for i2 in m.d2 for i_n in m.dn))
            ref_data.update((self._hashRef(Reference(m.v_1n2n[:, i_na, i2, i_nb])) for i_na in m.dn for i2 in m.d2 for i_nb in m.dn))
            ref_data.update((self._hashRef(Reference(m.b[:, i2, i_n].v0)) for i2 in m.d2 for i_n in m.dn))
            ref_data.update((self._hashRef(Reference(m.b[:, i2a, i_n].v2[i2b])) for i2a in m.d2 for i_n in m.dn for i2b in m.d2))
            ref_data.update((self._hashRef(Reference(m.b[:, i2, i_na].vn[i_nb])) for i2 in m.d2 for i_na in m.dn for i_nb in m.dn))
            assert len(ref_data) == len(comps)
            assert len(ref_data) == 38
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 1 and sets[0] is UnindexedComponent_set:
            ref_data = set()
            ref_data.update((self._hashRef(v) for v in m.v_2n.values()))
            assert len(ref_data) == len(comps)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 2 and sets[0] is m.d1 and (sets[1] is m.d1):
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.b[:, i2, i_n].v1[:])) for i2 in m.d2 for i_n in m.dn))
            assert len(ref_data) == len(comps)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)