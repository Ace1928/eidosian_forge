import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_flatten_m1_along_time(self):
    m = self._model1_1d_sets()
    sets = ComponentSet((m.time,))
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var)
    S = m.space
    SS = m.space * m.space
    SC = m.space * m.comp
    SSC = m.space * m.space * m.comp
    assert len(sets_list) == 3
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is UnindexedComponent_set:
            ref_data = {self._hashRef(Reference(m.v0))}
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 1 and sets[0] is m.time:
            ref_data = {self._hashRef(Reference(m.v1)), self._hashRef(Reference(m.b.b1[:].v0))}
            ref_data.update((self._hashRef(Reference(m.v2[:, x])) for x in S))
            ref_data.update((self._hashRef(Reference(m.v3[:, x, j])) for x, j in SC))
            ref_data.update((self._hashRef(Reference(m.b.b1[:].v1[x])) for x in S))
            ref_data.update((self._hashRef(Reference(m.b.b1[:].v2[x, j])) for x, j in SC))
            ref_data.update((self._hashRef(Reference(m.b.b1[:].b_s[x].v0)) for x in S))
            ref_data.update((self._hashRef(Reference(m.b.b1[:].b_s[x1].v1[x2])) for x1, x2 in SS))
            ref_data.update((self._hashRef(Reference(m.b.b1[:].b_s[x1].v2[x2, j])) for x1, x2, j in SSC))
            ref_data.update((self._hashRef(Reference(m.b.b2[:, x].v0)) for x in S))
            ref_data.update((self._hashRef(Reference(m.b.b2[:, x].v1[j])) for x, j in SC))
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        elif len(sets) == 2 and sets[0] is m.time and (sets[1] is m.time):
            ref_data = {self._hashRef(Reference(m.v_tt))}
            ref_data.update((self._hashRef(Reference(m.v_tst[:, x, :])) for x in S))
            ref_data.update((self._hashRef(Reference(m.b.b2[:, x].v2[:, j])) for x, j in SC))
            assert len(comps) == len(ref_data)
            for comp in comps:
                self.assertIn(self._hashRef(comp), ref_data)
        else:
            raise RuntimeError()