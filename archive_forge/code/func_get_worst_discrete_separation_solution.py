from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.objective import Objective, maximize, value
from pyomo.core.base import Var, Param
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.dependencies import numpy as np
from pyomo.contrib.pyros.util import ObjectiveType, get_time_from_solver
from pyomo.contrib.pyros.solve_data import (
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, is_certain_parameter
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.common.errors import ApplicationError
from pyomo.contrib.pyros.util import ABS_CON_CHECK_FEAS_TOL
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import (
import os
from copy import deepcopy
from itertools import product
def get_worst_discrete_separation_solution(performance_constraint, model_data, config, perf_cons_to_evaluate, discrete_solve_results):
    """
    Determine separation solution (and therefore worst-case
    uncertain parameter realization) with maximum violation
    of specified performance constraint.

    Parameters
    ----------
    performance_constraint : Constraint
        Performance constraint of interest.
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        User-specified PyROS solver settings.
    perf_cons_to_evaluate : list of Constraint
        Performance constraints for which to report violations
        by separation solution.
    discrete_solve_results : DiscreteSeparationSolveCallResults
        Separation problem solutions corresponding to the
        uncertain parameter scenarios listed in
        ``config.uncertainty_set.scenarios``.

    Returns
    -------
    SeparationSolveCallResult
        Solver call result for performance constraint of interest.
    """
    violations_of_perf_con = [solve_call_res.scaled_violations[performance_constraint] for solve_call_res in discrete_solve_results.solver_call_results.values()]
    list_of_scenario_idxs = list(discrete_solve_results.solver_call_results.keys())
    worst_case_res = discrete_solve_results.solver_call_results[list_of_scenario_idxs[np.argmax(violations_of_perf_con)]]
    worst_case_violation = np.max(violations_of_perf_con)
    assert worst_case_violation in worst_case_res.scaled_violations.values()
    eval_perf_con_scaled_violations = ComponentMap(((perf_con, worst_case_res.scaled_violations[perf_con]) for perf_con in perf_cons_to_evaluate))
    is_optimized_performance_con = performance_constraint is discrete_solve_results.performance_constraint
    if is_optimized_performance_con:
        results_list = [res for solve_call_results in discrete_solve_results.solver_call_results.values() for res in solve_call_results.results_list]
    else:
        results_list = []
    return SeparationSolveCallResults(solved_globally=worst_case_res.solved_globally, results_list=results_list, scaled_violations=eval_perf_con_scaled_violations, violating_param_realization=worst_case_res.violating_param_realization, variable_values=worst_case_res.variable_values, found_violation=worst_case_violation > config.robust_feasibility_tolerance, time_out=False, subsolver_error=False, discrete_set_scenario_index=worst_case_res.discrete_set_scenario_index)