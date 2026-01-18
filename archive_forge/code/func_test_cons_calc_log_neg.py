import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.environ as pyo
from pyomo.contrib.viewer.model_browser import ComponentDataItem
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.dependencies import DeferredImportError
def test_cons_calc_log_neg(self):
    cdi = ComponentDataItem(parent=None, ui_data=UIData(model=self.m), o=self.m.c6)
    cdi.ui_data.calculate_constraints()
    self.assertIsNone(cdi.get('value'))