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
def get_argmax_sum_violations(solver_call_results_map, perf_cons_to_evaluate):
    """
    Get key of entry of `solver_call_results_map` which contains
    separation problem solution with maximal sum of performance
    constraint violations over a specified sequence of performance
    constraints.

    Parameters
    ----------
    solver_call_results : ComponentMap
        Mapping from performance constraints to corresponding
        separation solver call results.
    perf_cons_to_evaluate : list of Constraints
        Performance constraints to consider for evaluating
        maximal sum.

    Returns
    -------
    worst_perf_con : None or Constraint
        Performance constraint corresponding to solver call
        results object containing solution with maximal sum
        of violations across all performance constraints.
        If ``found_violation`` attribute of all value entries of
        `solver_call_results_map` is False, then `None` is
        returned, as this means none of the performance constraints
        were found to be violated.
    """
    idx_to_perf_con_map = {idx: perf_con for idx, perf_con in enumerate(solver_call_results_map)}
    idxs_of_violated_cons = [idx for idx, perf_con in idx_to_perf_con_map.items() if solver_call_results_map[perf_con].found_violation]
    num_violated_cons = len(idxs_of_violated_cons)
    if num_violated_cons == 0:
        return None
    violations_arr = np.zeros(shape=(num_violated_cons, num_violated_cons))
    idxs_product = product(enumerate(idxs_of_violated_cons), enumerate(idxs_of_violated_cons))
    for (row_idx, viol_con_idx), (col_idx, viol_param_idx) in idxs_product:
        violations_arr[row_idx, col_idx] = max(0, solver_call_results_map[idx_to_perf_con_map[viol_param_idx]].scaled_violations[idx_to_perf_con_map[viol_con_idx]])
    worst_col_idx = np.argmax(np.sum(violations_arr, axis=0))
    return idx_to_perf_con_map[idxs_of_violated_cons[worst_col_idx]]