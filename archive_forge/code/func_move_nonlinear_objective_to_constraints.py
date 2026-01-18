from contextlib import contextmanager
import logging
from math import fabs
import sys
from pyomo.common import timing
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (
from pyomo.core.expr.numvalue import native_types
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.opt import SolverFactory
def move_nonlinear_objective_to_constraints(util_block, logger):
    m = util_block.parent_block()
    discrete_obj = next(m.component_data_objects(Objective, descend_into=True, active=True))
    if discrete_obj.polynomial_degree() in (1, 0):
        return None
    logger.info('Objective is nonlinear. Moving it to constraint set.')
    util_block.objective_value = Var(domain=Reals, initialize=0)
    if mcpp_available():
        mc_obj = McCormick(discrete_obj.expr)
        util_block.objective_value.setub(mc_obj.upper())
        util_block.objective_value.setlb(mc_obj.lower())
    else:
        lb, ub = compute_bounds_on_expr(discrete_obj.expr)
        if discrete_obj.sense == minimize:
            util_block.objective_value.setlb(lb)
        else:
            util_block.objective_value.setub(ub)
    if discrete_obj.sense == minimize:
        util_block.objective_constr = Constraint(expr=util_block.objective_value >= discrete_obj.expr)
    else:
        util_block.objective_constr = Constraint(expr=util_block.objective_value <= discrete_obj.expr)
    discrete_obj.deactivate()
    util_block.objective = Objective(expr=util_block.objective_value, sense=discrete_obj.sense)
    util_block.algebraic_variable_list.append(util_block.objective_value)
    if hasattr(util_block, 'constraint_list'):
        util_block.constraint_list.append(util_block.objective_constr)
    return discrete_obj