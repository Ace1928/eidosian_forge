import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def test_from_block(self):
    m = self.model()
    comp = m.v0
    pred_stack = [(self.get_attribute, 'v0')]
    self.assertCorrectStack(comp, pred_stack, context=m)
    comp = m.b.b1[2].b.b2[1, 'a', 1]
    pred_stack = [(self.get_item, (1, 'a', 1)), (self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 2), (self.get_attribute, 'b1'), (self.get_attribute, 'b')]
    self.assertCorrectStack(comp, pred_stack, context=m)
    comp = m.b.b1[2].b.b2[1, 'a', 1].b.v2['b', 2]
    pred_stack = [(self.get_item, ('b', 2)), (self.get_attribute, 'v2'), (self.get_attribute, 'b'), (self.get_item, (1, 'a', 1)), (self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 2)]
    self.assertCorrectStack(comp, pred_stack, context=m.b.b1)
    comp = m.b.b1[2].b.b2
    pred_stack = [(self.get_attribute, 'b2'), (self.get_attribute, 'b'), (self.get_item, 2)]
    self.assertCorrectStack(comp, pred_stack, context=m.b.b1)
    comp = m.b.b1[2]
    pred_stack = [(self.get_item, 2)]
    self.assertCorrectStack(comp, pred_stack, context=m.b.b1)
    comp = m.b.b1
    act_stack = get_component_call_stack(comp, context=m.b.b1)
    self.assertEqual(len(act_stack), 0)