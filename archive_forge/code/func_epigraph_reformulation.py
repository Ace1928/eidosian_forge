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
def epigraph_reformulation(exp, slack_var_list, constraint_list, use_mcpp, sense):
    """Epigraph reformulation.

    Generate the epigraph reformulation for objective expressions.

    Parameters
    ----------
    slack_var_list : VarList
        Slack vars for epigraph reformulation.
    constraint_list : ConstraintList
        Epigraph constraint list.
    use_mcpp : Bool
        Whether to use mcpp to tighten the bound of slack variables.
    exp : Expression
        The expression to reformulate.
    sense : objective sense
        The objective sense.
    """
    slack_var = slack_var_list.add()
    if mcpp_available() and use_mcpp:
        mc_obj = McCormick(exp)
        slack_var.setub(mc_obj.upper())
        slack_var.setlb(mc_obj.lower())
    else:
        lb, ub = compute_bounds_on_expr(exp)
        if sense == minimize:
            slack_var.setlb(lb)
        else:
            slack_var.setub(ub)
    if sense == minimize:
        constraint_list.add(expr=slack_var >= exp)
    else:
        constraint_list.add(expr=slack_var <= exp)