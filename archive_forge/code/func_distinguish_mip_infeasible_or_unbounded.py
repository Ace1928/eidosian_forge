from copy import deepcopy
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.util import (
from pyomo.core import Objective, Constraint
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
def distinguish_mip_infeasible_or_unbounded(m, config):
    """Distinguish between an infeasible or unbounded solution.

    Linear solvers will sometimes tell me that a problem is infeasible or
    unbounded during presolve, but not distinguish between the two cases. We
    address this by solving again with a solver option flag on.

    """
    tmp_args = deepcopy(config.mip_solver_args)
    if config.mip_solver == 'gurobi':
        tmp_args['options'] = tmp_args.get('options', {})
        tmp_args['options']['DualReductions'] = 0
    mipopt = SolverFactory(config.mip_solver)
    if isinstance(mipopt, PersistentSolver):
        mipopt.set_instance(m)
    with SuppressInfeasibleWarning():
        results = mipopt.solve(m, load_solutions=False, **tmp_args)
        if len(results.solution) > 0:
            m.solutions.load_from(results)
    termination_condition = results.solver.termination_condition
    return (results, termination_condition)