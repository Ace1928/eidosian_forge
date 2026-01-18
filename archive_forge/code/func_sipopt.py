from pyomo.environ import (
from pyomo.common.sorting import sorted_robust
from pyomo.core.expr import ExpressionReplacementVisitor
from pyomo.common.modeling import unique_component_name
from pyomo.common.deprecation import deprecated
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.contrib.sensitivity_toolbox.k_aug import K_augInterface, InTempDir
import logging
import os
import io
import shutil
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy, scipy_available
@deprecated("The sipopt function has been deprecated. Use the sensitivity_calculation() function with method='sipopt' to access this functionality.", logger='pyomo.contrib.sensitivity_toolbox', version='6.1')
def sipopt(instance, paramSubList, perturbList, cloneModel=True, tee=False, keepfiles=False, streamSoln=False):
    m = sensitivity_calculation('sipopt', instance, paramSubList, perturbList, cloneModel, tee, keepfiles, solver_options=None)
    return m