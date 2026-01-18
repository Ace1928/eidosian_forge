import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_cuids_some_sets_no_subblocks(self):
    m = ConcreteModel()
    m.s1 = Set(initialize=[1, 2, 3])
    m.s2 = Set(initialize=['a', 'b'])
    m.s3 = Set(initialize=[4, 5, 6])
    m.s4 = Set(initialize=['c', 'd'])
    m.v1 = Var(m.s1, m.s4)
    pred_cuid_set = {'v1[1,*]', 'v1[2,*]', 'v1[3,*]'}
    sets = (m.s3, m.s4)
    ctype = Var
    sets_list, comps_list = flatten_components_along_sets(m, sets, ctype)
    for sets, comps in zip(sets_list, comps_list):
        if len(sets) == 1 and sets[0] is m.s4:
            self.assertEqual(len(comps), len(m.s1))
            cuid_set = set((str(ComponentUID(comp.referent)) for comp in comps))
            self.assertEqual(cuid_set, pred_cuid_set)
        else:
            raise RuntimeError()