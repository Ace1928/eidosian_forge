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
def setup_sensitivity(self, paramList):
    instance = self.model_instance
    paramList = self._process_param_list(paramList)
    existing_block = instance.component(self.get_default_block_name())
    block = self._add_data_block(existing_block=existing_block)
    block._sens_data_list = []
    block._paramList = paramList
    self._add_sensitivity_data(paramList)
    sens_data_list = block._sens_data_list
    for var, _, _, _ in sens_data_list:
        var.unfix()
    variableSubMap = dict(((id(param), var) for var, param, list_idx, _ in sens_data_list if paramList[list_idx].ctype is Param))
    if variableSubMap:
        block._replaced_map = self._replace_parameters_in_constraints(variableSubMap)
        block._has_replaced_expressions = True
    block.paramConst = ConstraintList()
    for var, param, _, _ in sens_data_list:
        block.paramConst.add(var - param == 0)
    _add_sensitivity_suffixes(instance)
    for i, (var, _, _, _) in enumerate(sens_data_list):
        idx = i + 1
        con = block.paramConst[idx]
        instance.sens_state_0[var] = idx
        instance.sens_state_1[var] = idx
        instance.sens_init_constr[con] = idx
        instance.dcdp[con] = idx