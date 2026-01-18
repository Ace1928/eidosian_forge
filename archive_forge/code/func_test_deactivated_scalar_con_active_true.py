import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_deactivated_scalar_con_active_true(self):
    m = ConcreteModel()
    m.time = Set(initialize=[0, 1, 2])
    m.comp = Set(initialize=['A', 'B'])
    m.v = Var()

    def c_rule(m, j):
        return m.v == 1
    m.c = Constraint(m.comp, rule=c_rule)
    m.c[:].deactivate()
    sets = (m.time,)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Constraint, active=True)
    self.assertEqual(len(sets_list), 0)
    self.assertEqual(len(comps_list), 0)