import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def assertSameMap(self, pred, act):
    self.assertEqual(len(pred), len(act))
    for k, v in pred.items():
        self.assertIn(k, act)
        self.assertIs(pred[k], act[k])