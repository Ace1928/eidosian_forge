import os
import logging
import pyomo.contrib.viewer.report as rpt
import pyomo.environ as pyo
import pyomo.contrib.viewer.qt as myqt
from pyomo.contrib.viewer.model_browser import ModelBrowser
from pyomo.contrib.viewer.residual_table import ResidualTable
from pyomo.contrib.viewer.model_select import ModelSelect
from pyomo.contrib.viewer.ui_data import UIData
from pyomo.common.fileutils import this_file_dir
def constraints_restart(self):
    self.constraints = self._tree_restart(w=self.constraints, standard='Constraint', ui_data=self.ui_data)