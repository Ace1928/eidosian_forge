import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
@staticmethod
def nonnegativeBounds(var):
    if var.lb is not None and var.lb >= 0:
        return True
    elif var.domain is not None and var.domain.bounds()[0] >= 0:
        return True
    else:
        return False