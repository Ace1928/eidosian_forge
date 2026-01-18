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
def set_solver_constraint_violation_tolerance(opt, solver_name, config, warm_start=True):
    """Set constraint violation tolerance for solvers.

    Parameters
    ----------
    opt : Solvers
        The solver object.
    solver_name : String
        The name of solver.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if solver_name == 'baron':
        opt.options['AbsConFeasTol'] = config.zero_tolerance
    elif solver_name in {'ipopt', 'appsi_ipopt'}:
        opt.options['constr_viol_tol'] = config.zero_tolerance
    elif solver_name == 'cyipopt':
        opt.config.options['constr_viol_tol'] = config.zero_tolerance
    elif solver_name == 'gams':
        if config.nlp_solver_args['solver'] in {'ipopt', 'ipopth', 'msnlp', 'conopt', 'baron'}:
            opt.options['add_options'].append('GAMS_MODEL.optfile=1')
            opt.options['add_options'].append('$onecho > ' + config.nlp_solver_args['solver'] + '.opt')
            if config.nlp_solver_args['solver'] in {'ipopt', 'ipopth'}:
                opt.options['add_options'].append('constr_viol_tol ' + str(config.zero_tolerance))
                if warm_start:
                    opt.options['add_options'].append('warm_start_init_point       yes\nwarm_start_bound_push       1e-9\nwarm_start_bound_frac       1e-9\nwarm_start_slack_bound_frac 1e-9\nwarm_start_slack_bound_push 1e-9\nwarm_start_mult_bound_push  1e-9\n')
            elif config.nlp_solver_args['solver'] == 'conopt':
                opt.options['add_options'].append('RTNWMA ' + str(config.zero_tolerance))
            elif config.nlp_solver_args['solver'] == 'msnlp':
                opt.options['add_options'].append('feasibility_tolerance ' + str(config.zero_tolerance))
            elif config.nlp_solver_args['solver'] == 'baron':
                opt.options['add_options'].append('AbsConFeasTol ' + str(config.zero_tolerance))
            opt.options['add_options'].append('$offecho')