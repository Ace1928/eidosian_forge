import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_get_set_boolvar(self):
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.y)
    self.assertIsNone(cdi.set('value', True))
    self.assertEqual(pyo.value(self.m.y[1]), True)
    self.assertIsNone(cdi.set('value', False))
    self.assertEqual(pyo.value(self.m.y[1]), False)