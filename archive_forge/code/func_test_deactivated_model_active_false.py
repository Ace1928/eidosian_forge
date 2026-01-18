import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
def test_deactivated_model_active_false(self):
    m = self._model1_1d_sets()
    m.deactivate()
    sets = (m.time,)
    sets_list, comps_list = flatten_components_along_sets(m, sets, Var, active=True)
    self.assertEqual(len(sets_list), 0)
    self.assertEqual(len(comps_list), 0)