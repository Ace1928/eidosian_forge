from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
import pyomo.core.expr as EXPR
from pyomo.core.base.objective import Objective
from pyomo.opt.results.solver import (
from pyomo.contrib.solver.results import TerminationCondition, SolutionStatus
def assert_optimal_termination(results):
    """
    This function checks if the termination condition for the solver
    is 'optimal', 'locallyOptimal', or 'globallyOptimal', and the status is 'ok'
    and it raises a RuntimeError exception if this is not true.

    Parameters
    ----------
    results : Pyomo Results object returned from solver.solve
    """
    if not check_optimal_termination(results):
        if hasattr(results, 'solution_status'):
            msg = 'Solver failed to return an optimal solution. Solution status: {}, Termination condition: {}'.format(results.solution_status, results.termination_condition)
        else:
            msg = 'Solver failed to return an optimal solution. Solver status: {}, Termination condition: {}'.format(results.solver.status, results.solver.termination_condition)
        raise RuntimeError(msg)