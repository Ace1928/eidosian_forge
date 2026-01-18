import logging
from pyomo.common.collections import ComponentMap
from pyomo.contrib.viewer.qt import *
import pyomo.environ as pyo
def emit_exec_refresh(self):
    self.exec_refresh.emit()