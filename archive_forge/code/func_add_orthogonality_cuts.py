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
def add_orthogonality_cuts(working_model, mip_model, config):
    """Add orthogonality cuts.

    This function adds orthogonality cuts to avoid cycling when the independence constraint qualification is not satisfied.

    Parameters
    ----------
    working_model : Pyomo model
        The working model(original model).
    mip_model : Pyomo model
        The mip model.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    mip_integer_vars = mip_model.MindtPy_utils.discrete_variable_list
    nlp_integer_vars = working_model.MindtPy_utils.discrete_variable_list
    orthogonality_cut = sum(((nlp_v.value - mip_v.value) * (mip_v - nlp_v.value) for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars))) >= 0
    mip_model.MindtPy_utils.cuts.fp_orthogonality_cuts.add(orthogonality_cut)
    if config.fp_projcuts:
        orthogonality_cut = sum(((nlp_v.value - mip_v.value) * (nlp_v - nlp_v.value) for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars))) >= 0
        working_model.MindtPy_utils.cuts.fp_orthogonality_cuts.add(orthogonality_cut)