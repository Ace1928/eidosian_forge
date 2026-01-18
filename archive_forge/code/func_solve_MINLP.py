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
def solve_MINLP(util_block, config, timing):
    """Solve the MINLP subproblem."""
    config.logger.debug('Solving MINLP subproblem for fixed logical realizations.')
    model = util_block.parent_block()
    minlp_solver = SolverFactory(config.minlp_solver)
    if not minlp_solver.available():
        raise RuntimeError('MINLP solver %s is not available.' % config.minlp_solver)
    results = configure_and_call_solver(model, config.minlp_solver, config.minlp_solver_args, 'MINLP', timing, config.time_limit)
    subprob_termination = process_nonlinear_problem_results(results, model, 'MINLP', config)
    return subprob_termination