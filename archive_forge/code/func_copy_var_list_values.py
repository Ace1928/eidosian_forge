import logging
from pyomo.common.collections import ComponentMap
from pyomo.core import (
from pyomo.repn import generate_standard_repn
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import pyomo.core.expr as EXPR
from pyomo.opt import ProblemSense
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.util.model_size import build_model_size_report
from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import math
def copy_var_list_values(from_list, to_list, config, skip_stale=False, skip_fixed=True, ignore_integrality=False):
    """Copy variable values from one list to another.
    Rounds to Binary/Integer if necessary
    Sets to zero for NonNegativeReals if necessary

    from_list : list
        The variables that provide the values to copy from.
    to_list : list
        The variables that need to set value.
    config : ConfigBlock
        The specific configurations for MindtPy.
    skip_stale : bool, optional
        Whether to skip the stale variables, by default False.
    skip_fixed : bool, optional
        Whether to skip the fixed variables, by default True.
    ignore_integrality : bool, optional
        Whether to ignore the integrality of integer variables, by default False.
    """
    for v_from, v_to in zip(from_list, to_list):
        if skip_stale and v_from.stale:
            continue
        if skip_fixed and v_to.is_fixed():
            continue
        var_val = value(v_from, exception=False)
        set_var_valid_value(v_to, var_val, config.integer_tolerance, config.zero_tolerance, ignore_integrality)