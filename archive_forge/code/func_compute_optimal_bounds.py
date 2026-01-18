from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base.external import ExternalFunction
from pyomo.network import Port
from pyomo.common.collections import ComponentSet
from pyomo.repn import generate_standard_repn
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from weakref import ref as weakref_ref
from math import floor
import logging
def compute_optimal_bounds(expr, global_constraints, opt):
    """
    Returns a tuple (LB, UB) where LB and UB are the results of minimizing
    and maximizing expr over the variable bounds and the constraints on the
    global_constraints block. Note that if expr is nonlinear, even if one of
    the min and max problems is convex, the other won't be!

    Arguments:
    ----------
    expr : The subexpression whose bounds we will return
    global_constraints : A Block which contains the global Constraints and Vars
                         of the original model
    opt : A configured SolverFactory to use to minimize and maximize expr over
          the set defined by global_constraints. Note that if expr is nonlinear,
          opt will need to be capable of optimizing nonconvex problems.
    """
    if opt is None:
        raise GDP_Error("No solver was specified to optimize the subproblems for computing expression bounds! Please specify a configured solver in the 'compute_bounds_solver' argument if using 'compute_optimal_bounds.'")
    obj = Objective(expr=expr)
    global_constraints.add_component(unique_component_name(global_constraints, 'tmp_obj'), obj)
    results = opt.solve(global_constraints)
    if verify_successful_solve(results) is not NORMAL:
        logger.warning('Problem to find lower bound for expression %sdid not solve normally.\n\n%s' % (expr, results))
        LB = None
    else:
        LB = value(obj.expr)
    obj.sense = maximize
    results = opt.solve(global_constraints)
    if verify_successful_solve(results) is not NORMAL:
        logger.warning('Problem to find upper bound for expression %sdid not solve normally.\n\n%s' % (expr, results))
        UB = None
    else:
        UB = value(obj.expr)
    global_constraints.del_component(obj)
    del obj
    return (LB, UB)