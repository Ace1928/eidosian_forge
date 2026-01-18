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
def generate_norm2sq_objective_function(model, setpoint_model, discrete_only=False):
    """This function generates objective (FP-NLP subproblem) for minimum
    euclidean distance to setpoint_model.

    L2 distance of (x,y) = \\sqrt{\\sum_i (x_i - y_i)^2}.

    Parameters
    ----------
    model : Pyomo model
        The model that needs new objective function.
    setpoint_model : Pyomo model
        The model that provides the base point for us to calculate the distance.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete
        variables, by default False.

    Returns
    -------
    Objective
        The norm2 square objective function.
    """
    var_filter = (lambda v: v[1].is_integer()) if discrete_only else lambda v: 'MindtPy_utils.objective_value' not in v[1].name and 'MindtPy_utils.feas_opt.slack_var' not in v[1].name
    model_vars, setpoint_vars = zip(*filter(var_filter, zip(model.MindtPy_utils.variable_list, setpoint_model.MindtPy_utils.variable_list)))
    assert len(model_vars) == len(setpoint_vars), 'Trying to generate Squared Norm2 objective function for models with different number of variables'
    return Objective(expr=sum([(model_var - setpoint_var.value) ** 2 for model_var, setpoint_var in zip(model_vars, setpoint_vars)]))