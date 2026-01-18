from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
import pyomo.core.expr as EXPR
from pyomo.core.base.objective import Objective
from pyomo.opt.results.solver import (
from pyomo.contrib.solver.results import TerminationCondition, SolutionStatus
def check_optimal_termination(results):
    """
    This function returns True if the termination condition for the solver
    is 'optimal'.

    Parameters
    ----------
    results : Pyomo Results object returned from solver.solve

    Returns
    -------
    `bool`
    """
    if hasattr(results, 'solution_status'):
        if results.solution_status == SolutionStatus.optimal and results.termination_condition == TerminationCondition.convergenceCriteriaSatisfied:
            return True
    elif results.solver.status == SolverStatus.ok and (results.solver.termination_condition == LegacyTerminationCondition.optimal or results.solver.termination_condition == LegacyTerminationCondition.locallyOptimal or results.solver.termination_condition == LegacyTerminationCondition.globallyOptimal):
        return True
    return False