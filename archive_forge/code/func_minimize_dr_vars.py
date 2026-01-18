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
def minimize_dr_vars(model_data, config):
    """
    Polish decision rule of most recent master problem solution.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    results : SolverResults
        Subordinate solver results for the polishing problem.
    polishing_successful : bool
        True if polishing model was solved to acceptable level,
        False otherwise.
    """
    polishing_model = construct_dr_polishing_problem(model_data=model_data, config=config)
    if config.solve_master_globally:
        solver = config.global_solver
    else:
        solver = config.local_solver
    config.progress_logger.debug('Solving DR polishing problem')
    polishing_obj = polishing_model.polishing_obj
    config.progress_logger.debug(f' Initial DR norm: {value(polishing_obj)}')
    timer = TicTocTimer()
    orig_setting, custom_setting_present = adjust_solver_time_settings(model_data.timing, solver, config)
    model_data.timing.start_timer('main.dr_polishing')
    timer.tic(msg=None)
    try:
        results = solver.solve(polishing_model, tee=config.tee, load_solutions=False)
    except ApplicationError:
        config.progress_logger.error(f'Optimizer {repr(solver)} encountered an exception attempting to solve decision rule polishing problem in iteration {model_data.iteration}')
        raise
    else:
        setattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR, timer.toc(msg=None))
        model_data.timing.stop_timer('main.dr_polishing')
    finally:
        revert_solver_max_time_adjustment(solver, orig_setting, custom_setting_present, config)
    config.progress_logger.debug(' Done solving DR polishing problem')
    config.progress_logger.debug(f'  Termination condition: {results.solver.termination_condition} ')
    config.progress_logger.debug(f'  Solve time: {getattr(results.solver, TIC_TOC_SOLVE_TIME_ATTR)} s')
    acceptable = {tc.globallyOptimal, tc.optimal, tc.locallyOptimal}
    if results.solver.termination_condition not in acceptable:
        config.progress_logger.warning(f'Could not successfully solve DR polishing problem of iteration {model_data.iteration} with primary subordinate {('global' if config.solve_master_globally else 'local')} solver to acceptable level. Termination stats:\n{results.solver}\nMaintaining unpolished master problem solution.')
        return (results, False)
    polishing_model.solutions.load_from(results)
    for idx, blk in model_data.master_model.scenarios.items():
        ssv_zip = zip(blk.util.second_stage_variables, polishing_model.scenarios[idx].util.second_stage_variables)
        sv_zip = zip(blk.util.state_vars, polishing_model.scenarios[idx].util.state_vars)
        for master_ssv, polish_ssv in ssv_zip:
            master_ssv.set_value(value(polish_ssv))
        for master_sv, polish_sv in sv_zip:
            master_sv.set_value(value(polish_sv))
        dr_var_zip = zip(blk.util.decision_rule_vars, polishing_model.scenarios[idx].util.decision_rule_vars)
        for master_dr, polish_dr in dr_var_zip:
            for mvar, pvar in zip(master_dr.values(), polish_dr.values()):
                mvar.set_value(value(pvar), skip_validation=True)
    config.progress_logger.debug(f' Optimized DR norm: {value(polishing_obj)}')
    config.progress_logger.debug(' Polished master objective:')
    if config.objective_focus == ObjectiveType.worst_case:
        eval_obj_blk_idx = max(model_data.master_model.scenarios.keys(), key=lambda idx: value(model_data.master_model.scenarios[idx].second_stage_objective))
    else:
        eval_obj_blk_idx = (0, 0)
    eval_obj_blk = model_data.master_model.scenarios[eval_obj_blk_idx]
    config.progress_logger.debug(f'  First-stage objective: {value(eval_obj_blk.first_stage_objective)}')
    config.progress_logger.debug(f'  Second-stage objective: {value(eval_obj_blk.second_stage_objective)}')
    polished_master_obj = value(eval_obj_blk.first_stage_objective + eval_obj_blk.second_stage_objective)
    config.progress_logger.debug(f'  Objective: {polished_master_obj}')
    return (results, True)