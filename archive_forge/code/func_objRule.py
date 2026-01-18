import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import check_available_solvers
from pyomo.environ import (
from pyomo.core.plugins.transform.standard_form import StandardForm
from pyomo.core.plugins.transform.nonnegative_transform import NonNegativeTransformation
def objRule(model):
    return sum((5 * sum_product(model.__getattribute__(c + n)) for c in ('x', 'y', 'z') for n in ('1', '2', '3', '4')))