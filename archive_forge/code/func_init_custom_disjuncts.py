from contextlib import contextmanager
from math import fabs
from pyomo.common.collections import ComponentMap
from pyomo.contrib.gdpopt.cut_generation import add_no_good_cut
from pyomo.contrib.gdpopt.solve_discrete_problem import solve_MILP_discrete_problem
from pyomo.contrib.gdpopt.util import _DoNothing
from pyomo.core import Block, Constraint, Objective, Var, maximize, value
from pyomo.gdp import Disjunct
from pyomo.opt import TerminationCondition as tc
def init_custom_disjuncts(util_block, discrete_problem_util_block, subprob_util_block, config, solver):
    """Initialize by using user-specified custom disjuncts."""
    solver._log_header(config.logger)
    used_disjuncts = {}
    if config.mip_presolve:
        original_bounds = _collect_original_bounds(discrete_problem_util_block)
    for count, active_disjunct_set in enumerate(config.custom_init_disjuncts):
        used_disjuncts = set()
        subproblem = subprob_util_block.parent_block()
        config.logger.info('Generating initial linear GDP approximation by solving subproblems with user-specified active disjuncts.')
        for orig_disj, discrete_problem_disj in zip(util_block.disjunct_list, discrete_problem_util_block.disjunct_list):
            if orig_disj in active_disjunct_set:
                used_disjuncts.add(orig_disj)
                discrete_problem_disj.indicator_var.fix(True)
            else:
                discrete_problem_disj.indicator_var.fix(False)
        unused = set(active_disjunct_set) - used_disjuncts
        if len(unused) > 0:
            config.logger.warning('The following disjuncts from the custom disjunct initialization set number %s were unused: %s\nThey may not be Disjunct objects or they may not be on the active subtree being solved.' % (count, ', '.join([disj.name for disj in unused])))
        with preserve_discrete_problem_feasible_region(discrete_problem_util_block, config, original_bounds):
            mip_termination = solve_MILP_discrete_problem(discrete_problem_util_block, solver, config)
        if mip_termination is not tc.infeasible:
            solver._fix_discrete_soln_solve_subproblem_and_add_cuts(discrete_problem_util_block, subprob_util_block, config)
            add_no_good_cut(discrete_problem_util_block, config)
        else:
            config.logger.error('MILP relaxation infeasible for user-specified custom initialization disjunct set %s. Skipping that set and continuing on.' % list((disj.name for disj in active_disjunct_set)))
        solver.initialization_iteration += 1