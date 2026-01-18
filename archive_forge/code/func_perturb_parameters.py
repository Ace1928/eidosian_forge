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
def perturb_parameters(self, perturbList):
    instance = self.model_instance
    sens_data_list = self.block._sens_data_list
    paramConst = self.block.paramConst
    if len(self.block._paramList) != len(perturbList):
        raise ValueError('Length of paramList argument does not equal length of perturbList')
    for i, (var, param, list_idx, comp_idx) in enumerate(sens_data_list):
        con = paramConst[i + 1]
        if comp_idx is _NotAnIndex:
            ptb = value(perturbList[list_idx])
        else:
            try:
                ptb = value(perturbList[list_idx][comp_idx])
            except TypeError:
                ptb = value(perturbList[list_idx])
        instance.sens_state_value_1[var] = ptb
        instance.DeltaP[con] = value(var - ptb)