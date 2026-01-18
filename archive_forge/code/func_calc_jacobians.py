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
def calc_jacobians(constraint_list, differentiate_mode):
    """Generates a map of jacobians for the variables in the model.

    This function generates a map of jacobians corresponding to the variables in the
    constraint list.

    Parameters
    ----------
    constraint_list : List
        The list of constraints to calculate Jacobians.
    differentiate_mode : String
        The differentiate mode to calculate Jacobians.
    """
    jacobians = ComponentMap()
    mode = EXPR.differentiate.Modes(differentiate_mode)
    for c in constraint_list:
        vars_in_constr = list(EXPR.identify_variables(c.body))
        jac_list = EXPR.differentiate(c.body, wrt_list=vars_in_constr, mode=mode)
        jacobians[c] = ComponentMap(((var, jac_wrt_var) for var, jac_wrt_var in zip(vars_in_constr, jac_list)))
    return jacobians