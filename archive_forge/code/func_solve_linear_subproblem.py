from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.errors import InfeasibleConstraintException, DeveloperError
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.contrib.gdpopt.util import (
from pyomo.core import Constraint, TransformationFactory, Objective, Block
import pyomo.core.expr as EXPR
from pyomo.opt import SolverFactory, SolverResults
from pyomo.opt import TerminationCondition as tc
def solve_linear_subproblem(subproblem, config, timing):
    results = configure_and_call_solver(subproblem, config.mip_solver, config.mip_solver_args, 'MIP', timing, config.time_limit)
    subprob_terminate_cond = results.solver.termination_condition
    if subprob_terminate_cond is tc.optimal:
        return tc.optimal
    elif subprob_terminate_cond is tc.infeasibleOrUnbounded:
        results, subprob_terminate_cond = distinguish_mip_infeasible_or_unbounded(subproblem, config)
    if subprob_terminate_cond is tc.infeasible:
        config.logger.debug('MILP subproblem was infeasible.')
        return tc.infeasible
    elif subprob_terminate_cond is tc.unbounded:
        config.logger.debug('MILP subproblem was unbounded.')
        return tc.unbounded
    else:
        raise ValueError('GDPopt unable to handle MIP subproblem termination condition of %s. Results: %s' % (subprob_terminate_cond, results))