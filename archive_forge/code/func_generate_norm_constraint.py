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
def generate_norm_constraint(fp_nlp_model, mip_model, config):
    """Generate the norm constraint for the FP-NLP subproblem.

    Parameters
    ----------
    fp_nlp_model : Pyomo model
        The feasibility pump NLP subproblem.
    mip_model : Pyomo model
        The mip_model model.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if config.fp_main_norm == 'L1':
        generate_norm1_norm_constraint(fp_nlp_model, mip_model, config, discrete_only=True)
    elif config.fp_main_norm == 'L2':
        fp_nlp_model.norm_constraint = Constraint(expr=sum(((nlp_var - mip_var.value) ** 2 - config.fp_norm_constraint_coef * (nlp_var.value - mip_var.value) ** 2 for nlp_var, mip_var in zip(fp_nlp_model.MindtPy_utils.discrete_variable_list, mip_model.MindtPy_utils.discrete_variable_list))) <= 0)
    elif config.fp_main_norm == 'L_infinity':
        fp_nlp_model.norm_constraint = ConstraintList()
        rhs = config.fp_norm_constraint_coef * max((nlp_var.value - mip_var.value for nlp_var, mip_var in zip(fp_nlp_model.MindtPy_utils.discrete_variable_list, mip_model.MindtPy_utils.discrete_variable_list)))
        for nlp_var, mip_var in zip(fp_nlp_model.MindtPy_utils.discrete_variable_list, mip_model.MindtPy_utils.discrete_variable_list):
            fp_nlp_model.norm_constraint.add(nlp_var - mip_var.value <= rhs)