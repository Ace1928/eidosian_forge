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
def solver_call_separation(model_data, config, solve_globally, perf_con_to_maximize, perf_cons_to_evaluate):
    """
    Invoke subordinate solver(s) on separation problem.

    Parameters
    ----------
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.
    solve_globally : bool
        True to solve separation problems globally,
        False to solve locally.
    perf_con_to_maximize : Constraint
        Performance constraint for which to solve separation problem.
        Informs the objective (constraint violation) to maximize.
    perf_cons_to_evaluate : list of Constraint
        Performance constraints whose expressions are to be
        evaluated at the separation problem solution
        obtained.

    Returns
    -------
    solve_call_results : pyros.solve_data.SeparationSolveCallResults
        Solve results for separation problem of interest.
    """
    objectives_map = model_data.separation_model.util.map_obj_to_constr
    separation_obj = objectives_map[perf_con_to_maximize]
    if solve_globally:
        solvers = [config.global_solver] + config.backup_global_solvers
    else:
        solvers = [config.local_solver] + config.backup_local_solvers
    solver_status_dict = {}
    nlp_model = model_data.separation_model
    con_name_repr = get_con_name_repr(separation_model=nlp_model, con=perf_con_to_maximize, with_orig_name=True, with_obj_name=True)
    solve_mode = 'global' if solve_globally else 'local'
    initialize_separation(perf_con_to_maximize, model_data, config)
    separation_obj.activate()
    solve_call_results = SeparationSolveCallResults(solved_globally=solve_globally, time_out=False, results_list=[], found_violation=False, subsolver_error=False)
    timer = TicTocTimer()
    for idx, opt in enumerate(solvers):
        if idx > 0:
            config.progress_logger.warning(f'Invoking backup solver {opt!r} (solver {idx + 1} of {len(solvers)}) for {solve_mode} separation of performance constraint {con_name_repr} in iteration {model_data.iteration}.')
        orig_setting, custom_setting_present = adjust_solver_time_settings(model_data.timing, opt, config)
        model_data.timing.start_timer(f'main.{solve_mode}_separation')
        timer.tic(msg=None)
        try:
            results = opt.solve(nlp_model, tee=config.tee, load_solutions=False, symbolic_solver_labels=True)
        except ApplicationError:
            adverb = 'globally' if solve_globally else 'locally'
            config.progress_logger.error(f'Optimizer {repr(opt)} ({idx + 1} of {len(solvers)}) encountered exception attempting to {adverb} solve separation problem for constraint {con_name_repr} in iteration {model_data.iteration}.')
            raise
        else:
            setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
            model_data.timing.stop_timer(f'main.{solve_mode}_separation')
        finally:
            revert_solver_max_time_adjustment(opt, orig_setting, custom_setting_present, config)
        solver_status_dict[str(opt)] = results.solver.termination_condition
        solve_call_results.results_list.append(results)
        elapsed = get_main_elapsed_time(model_data.timing)
        if config.time_limit:
            if elapsed >= config.time_limit:
                solve_call_results.time_out = True
                separation_obj.deactivate()
                return solve_call_results
        acceptable_conditions = globally_acceptable if solve_globally else locally_acceptable
        optimal_termination = solve_call_results.termination_acceptable(acceptable_conditions)
        if optimal_termination:
            nlp_model.solutions.load_from(results)
            solve_call_results.variable_values = ComponentMap()
            for var in nlp_model.util.second_stage_variables:
                solve_call_results.variable_values[var] = value(var)
            for var in nlp_model.util.state_vars:
                solve_call_results.variable_values[var] = value(var)
            solve_call_results.violating_param_realization, solve_call_results.scaled_violations, solve_call_results.found_violation = evaluate_performance_constraint_violations(model_data, config, perf_con_to_maximize, perf_cons_to_evaluate)
            separation_obj.deactivate()
            return solve_call_results
        else:
            config.progress_logger.debug(f'Solver {opt} ({idx + 1} of {len(solvers)}) failed for {solve_mode} separation of performance constraint {con_name_repr} in iteration {model_data.iteration}. Termination condition: {results.solver.termination_condition!r}.')
            config.progress_logger.debug(f'Results:\n{results.solver}')
    solve_call_results.subsolver_error = True
    save_dir = config.subproblem_file_directory
    serialization_msg = ''
    if save_dir and config.keepfiles:
        objective = separation_obj.name
        output_problem_path = os.path.join(save_dir, config.uncertainty_set.type + '_' + nlp_model.name + '_separation_' + str(model_data.iteration) + '_obj_' + objective + '.bar')
        nlp_model.write(output_problem_path, io_options={'symbolic_solver_labels': True})
        serialization_msg = f' For debugging, problem has been serialized to the file {output_problem_path!r}.'
    solve_call_results.message = f'Could not successfully solve separation problem of iteration {model_data.iteration} for performance constraint {con_name_repr} with any of the provided subordinate {solve_mode} optimizers. (Termination statuses: {[str(term_cond) for term_cond in solver_status_dict.values()]}.){serialization_msg}'
    config.progress_logger.warning(solve_call_results.message)
    separation_obj.deactivate()
    return solve_call_results