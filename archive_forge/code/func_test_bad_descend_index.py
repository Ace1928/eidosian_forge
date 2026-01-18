import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_bad_descend_index(self):
    m = ConcreteModel()
    m.s1 = Set(initialize=[1, 2, 3])
    m.s2 = Set(initialize=['a', 'b'])
    m.v = Var(m.s1, m.s2)

    def b_rule(b, i, j):
        b.v = Var()
    m.b = Block(m.s1, m.s2, rule=b_rule)
    sets = (m.s1, m.s2)
    ctype = Var
    indices = ComponentMap([(m.s1, 'b')])
    with self.assertRaisesRegex(ValueError, 'bad index'):
        sets_list, comps_list = flatten_components_along_sets(m, sets, ctype, indices=indices)