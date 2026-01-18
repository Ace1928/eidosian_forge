import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_get_attr_that_does_not_exist(self):
    cdi = ComponentDataItem(parent=None, ui_data=None, o=self.m.x[1])
    self.assertIsNone(cdi.get('test_val'))