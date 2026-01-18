import logging
from pyomo.common.collections import ComponentMap
from pyomo.contrib.viewer.qt import *
import pyomo.environ as pyo
def calculate_constraints(self):
    for o in self.model.component_data_objects(pyo.Constraint, active=True):
        try:
            self.value_cache[o] = pyo.value(o.body, exception=False)
        except ZeroDivisionError:
            self.value_cache[o] = 'Divide_by_0'
    self.emit_exec_refresh()