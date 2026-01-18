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
def set_solver_mipgap(opt, solver_name, config):
    """Set mipgap for subsolvers.

    Parameters
    ----------
    opt : Solvers
        The solver object.
    solver_name : String
        The name of solver.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if solver_name in {'cplex', 'cplex_persistent', 'gurobi', 'gurobi_persistent', 'appsi_gurobi', 'glpk'}:
        opt.options['mipgap'] = config.mip_solver_mipgap
    elif solver_name == 'appsi_cplex':
        opt.options['mip_tolerances_mipgap'] = config.mip_solver_mipgap
    elif solver_name == 'appsi_highs':
        opt.config.mip_gap = config.mip_solver_mipgap
    elif solver_name == 'gams':
        opt.options['add_options'].append('option optcr=%s;' % config.mip_solver_mipgap)