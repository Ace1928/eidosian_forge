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
def make_separation_problem(model_data, config):
    """
    Swap out uncertain param Param objects for Vars
    Add uncertainty set constraints and separation objectives
    """
    separation_model = model_data.original.clone()
    separation_model.del_component('coefficient_matching_constraints')
    separation_model.del_component('coefficient_matching_constraints_index')
    uncertain_params = separation_model.util.uncertain_params
    separation_model.util.uncertain_param_vars = param_vars = Var(range(len(uncertain_params)))
    map_new_constraint_list_to_original_con = ComponentMap()
    if config.objective_focus is ObjectiveType.worst_case:
        separation_model.util.zeta = Param(initialize=0, mutable=True)
        constr = Constraint(expr=separation_model.first_stage_objective + separation_model.second_stage_objective - separation_model.util.zeta <= 0)
        separation_model.add_component('epigraph_constr', constr)
    substitution_map = {}
    for idx, var in enumerate(list(param_vars.values())):
        param = uncertain_params[idx]
        var.set_value(param.value, skip_validation=True)
        substitution_map[id(param)] = var
    separation_model.util.new_constraints = constraints = ConstraintList()
    uncertain_param_set = ComponentSet(uncertain_params)
    for c in separation_model.component_data_objects(Constraint):
        if any((v in uncertain_param_set for v in identify_mutable_parameters(c.expr))):
            if c.equality:
                if c in separation_model.util.h_x_q_constraints:
                    c.deactivate()
                else:
                    constraints.add(replace_expressions(expr=c.lower, substitution_map=substitution_map) == replace_expressions(expr=c.body, substitution_map=substitution_map))
            elif c.lower is not None:
                constraints.add(replace_expressions(expr=c.lower, substitution_map=substitution_map) <= replace_expressions(expr=c.body, substitution_map=substitution_map))
            elif c.upper is not None:
                constraints.add(replace_expressions(expr=c.upper, substitution_map=substitution_map) >= replace_expressions(expr=c.body, substitution_map=substitution_map))
            else:
                raise ValueError('Unable to parse constraint for building the separation problem.')
            c.deactivate()
            map_new_constraint_list_to_original_con[constraints[constraints.index_set().last()]] = c
    separation_model.util.map_new_constraint_list_to_original_con = map_new_constraint_list_to_original_con
    make_separation_objective_functions(separation_model, config)
    add_uncertainty_set_constraints(separation_model, config)
    for c in separation_model.util.h_x_q_constraints:
        c.deactivate()
    return separation_model