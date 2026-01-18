from copy import deepcopy
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.util import (
from pyomo.core import Objective, Constraint
from pyomo.opt import SolutionStatus, SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
def solve_MILP_discrete_problem(util_block, solver, config):
    """Solves the linear GDP model and attempts to resolve solution issues.
    Returns one of TerminationCondition.optimal, TerminationCondition.feasible,
    TerminationCondition.infeasible, or TerminationCondition.unbounded.
    """
    timing = solver.timing
    m = util_block.parent_block()
    if config.mip_presolve:
        try:
            fbbt(m, integer_tol=config.integer_tolerance, deactivate_satisfied_constraints=False)
        except InfeasibleConstraintException as e:
            config.logger.debug('MIP preprocessing detected infeasibility:\n\t%s' % str(e))
            return tc.infeasible
    getattr(m, 'ipopt_zL_out', _DoNothing()).deactivate()
    getattr(m, 'ipopt_zU_out', _DoNothing()).deactivate()
    if not SolverFactory(config.mip_solver).available():
        raise RuntimeError('MIP solver %s is not available.' % config.mip_solver)
    config.call_before_discrete_problem_solve(solver, m, util_block)
    if config.call_before_master_solve is not _DoNothing:
        deprecation_warning("The 'call_before_master_solve' argument is deprecated. Please use the 'call_before_discrete_problem_solve' option to specify the callback.", version='6.4.2')
    with SuppressInfeasibleWarning():
        mip_args = dict(config.mip_solver_args)
        if config.time_limit is not None:
            elapsed = get_main_elapsed_time(timing)
            remaining = max(config.time_limit - elapsed, 1)
            if config.mip_solver == 'gams':
                mip_args['add_options'] = mip_args.get('add_options', [])
                mip_args['add_options'].append('option reslim=%s;' % remaining)
            elif config.mip_solver == 'multisolve':
                mip_args['time_limit'] = min(mip_args.get('time_limit', float('inf')), remaining)
        results = SolverFactory(config.mip_solver).solve(m, **mip_args)
    config.call_after_discrete_problem_solve(solver, m, util_block)
    if config.call_after_master_solve is not _DoNothing:
        deprecation_warning("The 'call_after_master_solve' argument is deprecated. Please use the 'call_after_discrete_problem_solve' option to specify the callback.", version='6.4.2')
    terminate_cond = results.solver.termination_condition
    if terminate_cond is tc.infeasibleOrUnbounded:
        results, terminate_cond = distinguish_mip_infeasible_or_unbounded(m, config)
    if terminate_cond is tc.unbounded:
        obj_bound = 1000000000000000.0
        config.logger.warning('Discrete problem was unbounded. Re-solving with arbitrary bound values of (-{0:.10g}, {0:.10g}) on the objective, in order to get a discrete solution. Check your initialization routine.'.format(obj_bound))
        discrete_objective = next(m.component_data_objects(Objective, active=True))
        util_block.objective_bound = Constraint(expr=(-obj_bound, discrete_objective.expr, obj_bound))
        with SuppressInfeasibleWarning():
            results = SolverFactory(config.mip_solver).solve(m, **config.mip_solver_args)
        del util_block.objective_bound
        if results.solver.termination_condition in {tc.optimal, tc.feasible, tc.locallyOptimal, tc.globallyOptimal}:
            return tc.unbounded
        else:
            raise RuntimeError('Unable to find a feasible solution for the unbounded MILP discrete problem by bounding the objective. Either check your discrete problem initialization, or add a bound on the discrete problem objective value that admits a feasible solution.')
    if terminate_cond is tc.optimal:
        return tc.optimal
    elif terminate_cond in {tc.locallyOptimal, tc.feasible}:
        return tc.feasible
    elif terminate_cond is tc.infeasible:
        config.logger.info('MILP discrete problem is now infeasible. GDPopt has explored or cut off all feasible discrete configurations.')
        return tc.infeasible
    elif terminate_cond is tc.maxTimeLimit:
        if len(results.solution) > 0:
            config.logger.info('Unable to optimize MILP discrete problem within time limit. Using current solver feasible solution.')
            return tc.feasible
        else:
            config.logger.info('Unable to optimize MILP discrete problem within time limit. No solution found. Treating as infeasible, but there are no guarantees.')
            return tc.infeasible
    elif terminate_cond is tc.other and results.solution.status is SolutionStatus.feasible:
        config.logger.info('MIP solver reported feasible solution to MILP discrete problem, but it is not guaranteed to be optimal.')
        return tc.feasible
    else:
        raise ValueError('GDPopt unable to handle MILP discrete problem termination condition of %s. Solver message: %s' % (terminate_cond, results.solver.message))