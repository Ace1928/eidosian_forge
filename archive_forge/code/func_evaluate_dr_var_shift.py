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
def evaluate_dr_var_shift(current_master_dr_var_vals, previous_master_dr_var_vals, first_iter_master_nom_ssv_vals, dr_var_to_ssv_map):
    """
    Evaluate decision rule variable "shift": the maximum relative
    difference between scaled decision rule (DR) variable expressions
    (terms in the DR equations) from the current
    and previous master iterations.

    Parameters
    ----------
    current_master_dr_var_vals : ComponentMap
        DR variable values from the current master
        iteration.
    previous_master_dr_var_vals : ComponentMap
        DR variable values from the previous master
        iteration.
    first_iter_master_nom_ssv_vals : ComponentMap
        Second-stage variable values (evaluated subject to the
        nominal uncertain parameter realization)
        from the first master iteration.
    dr_var_to_ssv_map : ComponentMap
        Mapping from each DR variable to the
        second-stage variable whose value is a function of the
        DR variable.

    Returns
    -------
    None
        Returned only if `current_master_dr_var_vals` is empty,
        which should occur only if the problem has no decision rule
        (or equivalently, second-stage) variables.
    float
        The maximum relative difference.
        Returned only if `current_master_dr_var_vals` is not empty.
    """
    if not current_master_dr_var_vals:
        return None
    else:
        return max((abs(current_master_dr_var_vals[drvar] - previous_master_dr_var_vals[drvar]) / max((1, abs(first_iter_master_nom_ssv_vals[dr_var_to_ssv_map[drvar]]))) for drvar in previous_master_dr_var_vals))