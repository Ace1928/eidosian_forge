import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def assertSameStack(self, stack1, stack2):
    for (call1, arg1), (call2, arg2) in zip(stack1, stack2):
        self.assertIs(call1, call2)
        self.assertEqual(arg1, arg2)