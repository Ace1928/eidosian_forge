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
def set_var_valid_value(var, var_val, integer_tolerance, zero_tolerance, ignore_integrality):
    """This function tries to set a valid value for variable with the given input.
    Rounds to Binary/Integer if necessary.
    Sets to zero for NonNegativeReals if necessary.

    Parameters
    ----------
    var : Var
        The variable that needs to set value.
    var_val : float
        The desired value to set for var.
    integer_tolerance: float
        Tolerance on integral values.
    zero_tolerance: float
        Tolerance on variable equal to zero.
    ignore_integrality : bool, optional
        Whether to ignore the integrality of integer variables, by default False.

    Raises
    ------
    ValueError
        Cannot successfully set the value to the variable.
    """
    var.stale = True
    rounded_val = int(round(var_val))
    if var_val in var.domain and (not (var.has_lb() and var_val < var.lb)) and (not (var.has_ub() and var_val > var.ub)):
        var.set_value(var_val)
    elif var.has_lb() and var_val < var.lb:
        var.set_value(var.lb)
    elif var.has_ub() and var_val > var.ub:
        var.set_value(var.ub)
    elif ignore_integrality and var.is_integer():
        var.set_value(var_val, skip_validation=True)
    elif var.is_integer() and math.fabs(var_val - rounded_val) <= integer_tolerance:
        var.set_value(rounded_val)
    elif abs(var_val) <= zero_tolerance and 0 in var.domain:
        var.set_value(0)
    else:
        raise ValueError('set_var_valid_value failed with variable {}, value = {} and rounded value = {}'.format(var.name, var_val, rounded_val))