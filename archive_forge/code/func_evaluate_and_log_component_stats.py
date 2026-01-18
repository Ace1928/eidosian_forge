from pyomo.core.base import Objective, ConstraintList, Var, Constraint, Block
from pyomo.opt.results import TerminationCondition
from pyomo.contrib.pyros import master_problem_methods, separation_problem_methods
from pyomo.contrib.pyros.solve_data import SeparationProblemData, MasterResult
from pyomo.contrib.pyros.uncertainty_sets import Geometry
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import get_main_elapsed_time, coefficient_matching
from pyomo.core.base import value
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.var import _VarData as VarData
from itertools import chain
from pyomo.common.dependencies import numpy as np
def evaluate_and_log_component_stats(model_data, separation_model, config):
    """
    Evaluate and log model component statistics.
    """
    IterationLogRecord.log_header_rule(config.progress_logger.info)
    config.progress_logger.info('Model statistics:')
    dr_var_set = ComponentSet(chain(*tuple((indexed_dr_var.values() for indexed_dr_var in model_data.working_model.util.decision_rule_vars))))
    first_stage_vars = [var for var in model_data.working_model.util.first_stage_variables if var not in dr_var_set]
    sep_model_epigraph_con = getattr(separation_model, 'epigraph_constr', None)
    has_epigraph_con = sep_model_epigraph_con is not None
    num_fsv = len(first_stage_vars)
    num_ssv = len(model_data.working_model.util.second_stage_variables)
    num_sv = len(model_data.working_model.util.state_vars)
    num_dr_vars = len(dr_var_set)
    num_vars = int(has_epigraph_con) + num_fsv + num_ssv + num_sv + num_dr_vars
    num_uncertain_params = len(model_data.working_model.util.uncertain_params)
    eq_cons = [con for con in model_data.working_model.component_data_objects(Constraint, active=True) if con.equality]
    dr_eq_set = ComponentSet(chain(*tuple((indexed_dr_eq.values() for indexed_dr_eq in model_data.working_model.util.decision_rule_eqns))))
    num_eq_cons = len(eq_cons)
    num_dr_cons = len(dr_eq_set)
    num_coefficient_matching_cons = len(getattr(model_data.working_model, 'coefficient_matching_constraints', []))
    num_other_eq_cons = num_eq_cons - num_dr_cons - num_coefficient_matching_cons
    new_sep_con_map = separation_model.util.map_new_constraint_list_to_original_con
    perf_con_set = ComponentSet((new_sep_con_map.get(con, con) for con in separation_model.util.performance_constraints))
    is_epigraph_con_first_stage = has_epigraph_con and sep_model_epigraph_con not in perf_con_set
    working_model_perf_con_set = ComponentSet((model_data.working_model.find_component(new_sep_con_map.get(con, con)) for con in separation_model.util.performance_constraints if con is not None))
    num_perf_cons = len(separation_model.util.performance_constraints)
    num_fsv_bounds = sum((int(var.lower is not None) + int(var.upper is not None) for var in first_stage_vars))
    ineq_con_set = [con for con in model_data.working_model.component_data_objects(Constraint, active=True) if not con.equality]
    num_fsv_ineqs = num_fsv_bounds + len([con for con in ineq_con_set if con not in working_model_perf_con_set]) + is_epigraph_con_first_stage
    num_ineq_cons = len(ineq_con_set) + has_epigraph_con + num_fsv_bounds
    config.progress_logger.info(f'{'  Number of variables'} : {num_vars}')
    config.progress_logger.info(f'{'    Epigraph variable'} : {int(has_epigraph_con)}')
    config.progress_logger.info(f'{'    First-stage variables'} : {num_fsv}')
    config.progress_logger.info(f'{'    Second-stage variables'} : {num_ssv}')
    config.progress_logger.info(f'{'    State variables'} : {num_sv}')
    config.progress_logger.info(f'{'    Decision rule variables'} : {num_dr_vars}')
    config.progress_logger.info(f'{'  Number of uncertain parameters'} : {num_uncertain_params}')
    config.progress_logger.info(f'{'  Number of constraints'} : {num_ineq_cons + num_eq_cons}')
    config.progress_logger.info(f'{'    Equality constraints'} : {num_eq_cons}')
    config.progress_logger.info(f'{'      Coefficient matching constraints'} : {num_coefficient_matching_cons}')
    config.progress_logger.info(f'{'      Decision rule equations'} : {num_dr_cons}')
    config.progress_logger.info(f'{'      All other equality constraints'} : {num_other_eq_cons}')
    config.progress_logger.info(f'{'    Inequality constraints'} : {num_ineq_cons}')
    config.progress_logger.info(f'{'      First-stage inequalities (incl. certain var bounds)'} : {num_fsv_ineqs}')
    config.progress_logger.info(f'{'      Performance constraints (incl. var bounds)'} : {num_perf_cons}')