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
def call_appropriate_subproblem_solver(subprob_util_block, solver, config):
    timing = solver.timing
    subprob = subprob_util_block.parent_block()
    config.call_before_subproblem_solve(solver, subprob, subprob_util_block)
    if not any((constr.body.polynomial_degree() not in (1, 0) for constr in subprob.component_data_objects(Constraint, active=True))):
        subprob_termination = solve_linear_subproblem(subprob, config, timing)
    else:
        unfixed_discrete_vars = detect_unfixed_discrete_vars(subprob)
        if config.force_subproblem_nlp and len(unfixed_discrete_vars) > 0:
            raise DeveloperError('Unfixed discrete variables found on the NLP subproblem.')
        elif len(unfixed_discrete_vars) == 0:
            subprob_termination = solve_NLP(subprob, config, timing)
        else:
            config.logger.debug('The following discrete variables are unfixed: %s\nProceeding by solving the subproblem as a MINLP.' % ', '.join([v.name for v in unfixed_discrete_vars]))
            subprob_termination = solve_MINLP(subprob_util_block, config, timing)
    config.call_after_subproblem_solve(solver, subprob, subprob_util_block)
    if subprob_termination in {tc.optimal, tc.feasible}:
        config.call_after_subproblem_feasible(solver, subprob, subprob_util_block)
    return subprob_termination