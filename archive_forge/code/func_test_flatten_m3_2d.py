import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flatten_m3_2d(self):
    m = self._model3_nd_sets_normalizeflatten()
    sets = ComponentSet((m.d2,))
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    assert len(sets_list) == len(comps_list)
    assert len(sets_list) == 2
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is m.d2:
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.v_2n[:, :, i_n])) for i_n in m.dn))
            ref_data.update((self._hashRef(Reference(m.v_12[i1, :, :])) for i1 in m.d1))
            ref_data.update((self._hashRef(Reference(m.v_12n[i1, :, :, i_n])) for i1 in m.d1 for i_n in m.dn))
            ref_data.update((self._hashRef(Reference(m.v_1n2n[i1, i_na, :, :, i_nb])) for i1 in m.d1 for i_na in m.dn for i_nb in m.dn))
            ref_data.update((self._hashRef(Reference(m.b[i1, :, :, i_n].v0)) for i1 in m.d1 for i_n in m.dn))
            ref_data.update((self._hashRef(Reference(m.b[i1a, :, :, i_n].v1[i1b])) for i1a in m.d1 for i_n in m.dn for i1b in m.d1))
            ref_data.update((self._hashRef(Reference(m.b[i1, :, :, i_na].vn[i_nb])) for i1 in m.d1 for i_na in m.dn for i_nb in m.dn))
            assert len(ref_data) == len(comps)
            assert len(ref_data) == 36
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 2 and sets[0] is m.d2 and (sets[1] is m.d2):
            ref_data = set()
            ref_data.update((self._hashRef(Reference(m.v_212[:, :, i1, :, :])) for i1 in m.d1))
            ref_data.update((self._hashRef(Reference(m.b[i1, :, :, i_n].v2[:, :])) for i1 in m.d1 for i_n in m.dn))
            assert len(ref_data) == len(comps)
            assert len(ref_data) == 6
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()