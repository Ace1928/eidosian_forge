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
def model_information(self):
    """
        Put some useful model information into a message box

        Displays:
        * number of active equality constraints
        * number of free variables in equality constraints
        * degrees of freedom

        Other things that could be added
        * number of deactivated equalities
        * number of active inequality constraints
        * number of deactivated inequality constratins
        * number of free variables not appearing in active constraints
        * number of fixed variables not appearing in active constraints
        * number of free variables not appearing in any constraints
        * number of fixed variables not appearing in any constraints
        * number of fixed variables appearing in constraints
        """
    active_eq = rpt.count_equality_constraints(self.ui_data.model)
    free_vars = rpt.count_free_variables(self.ui_data.model)
    cons = rpt.count_constraints(self.ui_data.model)
    dof = free_vars - active_eq
    if dof == 1:
        doftext = 'Degree'
    else:
        doftext = 'Degrees'
    msg = myqt.QMessageBox()
    msg.setStyleSheet('QLabel{min-width: 600px;}')
    self._dialog = msg
    msg.setWindowTitle('Model Information')
    msg.setText('{} -- Active Constraints\n{} -- Active Equalities\n{} -- Free Variables\n{} -- {} of Freedom'.format(cons, active_eq, free_vars, dof, doftext))
    msg.setStandardButtons(myqt.QMessageBox.Ok)
    msg.setModal(False)
    msg.show()