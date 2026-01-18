from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.objective import Objective, maximize, value
from pyomo.core.base import Var, Param
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pyros.util import ObjectiveType, get_time_from_solver
from pyomo.contrib.pyros.solve_data import (
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, is_certain_parameter
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.common.errors import ApplicationError
from pyomo.contrib.pyros.util import ABS_CON_CHECK_FEAS_TOL
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import (
import os
from copy import deepcopy
from itertools import product
def eval_master_violation(block_idx):
    """
        Evaluate violation of `perf_con` by variables of
        specified master block.
        """
    new_con_map = model_data.separation_model.util.map_new_constraint_list_to_original_con
    in_new_cons = perf_con_to_maximize in new_con_map
    if in_new_cons:
        sep_con = new_con_map[perf_con_to_maximize]
    else:
        sep_con = perf_con_to_maximize
    master_con = model_data.master_model.scenarios[block_idx, 0].find_component(sep_con)
    return value(master_con)