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
def evaluate_performance_constraint_violations(model_data, config, perf_con_to_maximize, perf_cons_to_evaluate):
    """
    Evaluate the inequality constraint function violations
    of the current separation model solution, and store the
    results in a given `SeparationResult` object.
    Also, determine whether the separation solution violates
    the inequality constraint whose body is the model's
    active objective.

    Parameters
    ----------
    model_data : SeparationProblemData
        Object containing the separation model.
    config : ConfigDict
        PyROS solver settings.
    perf_cons_to_evaluate : list of Constraint
        Performance constraints whose expressions are to
        be evaluated at the current separation problem
        solution.
        Exactly one of these constraints should be mapped
        to an active Objective in the separation model.

    Returns
    -------
    violating_param_realization : list of float
        Uncertain parameter realization corresponding to maximum
        constraint violation.
    scaled_violations : ComponentMap
        Mapping from performance constraints to be evaluated
        to their violations by the separation problem solution.
    constraint_violated : bool
        True if performance constraint mapped to active
        separation model Objective is violated (beyond tolerance),
        False otherwise

    Raises
    ------
    ValueError
        If `perf_cons_to_evaluate` does not contain exactly
        1 entry which can be mapped to an active Objective
        of ``model_data.separation_model``.
    """
    violating_param_realization = list((param.value for param in model_data.separation_model.util.uncertain_param_vars.values()))
    violations_by_sep_solution = get_sep_objective_values(model_data=model_data, config=config, perf_cons=perf_cons_to_evaluate)
    scaled_violations = ComponentMap()
    for perf_con, sep_sol_violation in violations_by_sep_solution.items():
        scaled_violation = sep_sol_violation / max(1, abs(model_data.nom_perf_con_violations[perf_con]))
        scaled_violations[perf_con] = scaled_violation
        if perf_con is perf_con_to_maximize:
            scaled_active_obj_violation = scaled_violation
    constraint_violated = scaled_active_obj_violation > config.robust_feasibility_tolerance
    return (violating_param_realization, scaled_violations, constraint_violated)