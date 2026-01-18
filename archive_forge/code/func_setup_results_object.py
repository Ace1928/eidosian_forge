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
def setup_results_object(results, model, config):
    """Record problem statistics for original model."""
    res = results
    prob = res.problem
    res.problem.name = model.name
    res.problem.number_of_nonzeros = None
    res.solver.termination_condition = None
    res.solver.message = None
    res.solver.user_time = None
    res.solver.wallclock_time = None
    res.solver.termination_message = None
    res.solver.name = 'MindtPy' + str(config.strategy)
    num_of = build_model_size_report(model)
    prob.number_of_constraints = num_of.activated.constraints
    prob.number_of_disjunctions = num_of.activated.disjunctions
    prob.number_of_variables = num_of.activated.variables
    prob.number_of_binary_variables = num_of.activated.binary_variables
    prob.number_of_continuous_variables = num_of.activated.continuous_variables
    prob.number_of_integer_variables = num_of.activated.integer_variables
    config.logger.info('Original model has %s constraints (%s nonlinear) and %s disjunctions, with %s variables, of which %s are binary, %s are integer, and %s are continuous.' % (num_of.activated.constraints, num_of.activated.nonlinear_constraints, num_of.activated.disjunctions, num_of.activated.variables, num_of.activated.binary_variables, num_of.activated.integer_variables, num_of.activated.continuous_variables))
    config.logger.info('{} is the initial strategy being used.\n'.format(config.init_strategy))
    config.logger.info(' ===============================================================================================')
    config.logger.info(' {:>9} | {:>15} | {:>15} | {:>12} | {:>12} | {:^7} | {:>7}\n'.format('Iteration', 'Subproblem Type', 'Objective Value', 'Primal Bound', 'Dual Bound', ' Gap ', 'Time(s)'))