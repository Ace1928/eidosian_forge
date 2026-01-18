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
def solve_continuous_problem(m, config):
    logger = config.logger
    logger.info('Problem has no discrete decisions.')
    obj = next(m.component_data_objects(Objective, active=True))
    if any((c.body.polynomial_degree() not in (1, 0) for c in m.component_data_objects(Constraint, active=True, descend_into=Block))) or obj.polynomial_degree() not in (1, 0):
        logger.info('Your model is an NLP (nonlinear program). Using NLP solver %s to solve.' % config.nlp_solver)
        results = SolverFactory(config.nlp_solver).solve(m, **config.nlp_solver_args)
        return results
    else:
        logger.info('Your model is an LP (linear program). Using LP solver %s to solve.' % config.mip_solver)
        results = SolverFactory(config.mip_solver).solve(m, **config.mip_solver_args)
        return results