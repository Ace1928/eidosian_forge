from pyomo.core.base import (
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverResults
from pyomo.core.expr import value
from pyomo.core.base.set_types import NonNegativeIntegers, NonNegativeReals
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, MasterResult
from pyomo.opt.results import check_optimal_termination
from pyomo.core.expr.visitor import replace_expressions, identify_variables
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core import TransformationFactory
import itertools as it
import os
from copy import deepcopy
from pyomo.common.errors import ApplicationError
from pyomo.common.modeling import unique_component_name
from pyomo.common.timing import TicTocTimer
from pyomo.contrib.pyros.util import TIC_TOC_SOLVE_TIME_ATTR, enforce_dr_degree
def solve_master_feasibility_problem(model_data, config):
    """
    Solve a slack variable-based feasibility model derived
    from the master problem. Initialize the master problem
    to the  solution found by the optimizer if solved successfully,
    or to the initial point provided to the solver otherwise.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Solver results.
    """
    model = construct_master_feasibility_problem(model_data, config)
    active_obj = next(model.component_data_objects(Objective, active=True))
    config.progress_logger.debug('Solving master feasibility problem')
    config.progress_logger.debug(f' Initial objective (total slack): {value(active_obj)}')
    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver
    timer = TicTocTimer()
    orig_setting, custom_setting_present = adjust_solver_time_settings(model_data.timing, solver, config)
    model_data.timing.start_timer('main.master_feasibility')
    timer.tic(msg=None)
    try:
        results = solver.solve(model, tee=config.tee, load_solutions=False)
    except ApplicationError:
        config.progress_logger.error(f'Optimizer {repr(solver)} encountered exception attempting to solve master feasibility problem in iteration {model_data.iteration}.')
        raise
    else:
        setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
        model_data.timing.stop_timer('main.master_feasibility')
    finally:
        revert_solver_max_time_adjustment(solver, orig_setting, custom_setting_present, config)
    feasible_terminations = {tc.optimal, tc.locallyOptimal, tc.globallyOptimal, tc.feasible}
    if results.solver.termination_condition in feasible_terminations:
        model.solutions.load_from(results)
        config.progress_logger.debug(f' Final objective (total slack): {value(active_obj)}')
        config.progress_logger.debug(f' Termination condition: {results.solver.termination_condition}')
        config.progress_logger.debug(f' Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)}s')
    else:
        config.progress_logger.warning(f'Could not successfully solve master feasibility problem of iteration {model_data.iteration} with primary subordinate {('global' if config.solve_master_globally else 'local')} solver to acceptable level. Termination stats:\n{results.solver}\nMaintaining unoptimized point for master problem initialization.')
    for master_var, feas_var in model_data.feasibility_problem_varmap:
        master_var.set_value(feas_var.value, skip_validation=True)
    return results