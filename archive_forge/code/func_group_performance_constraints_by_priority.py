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
def group_performance_constraints_by_priority(model_data, config):
    """
    Group model performance constraints by separation priority.

    Parameters
    ----------
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        User-specified PyROS solve options.

    Returns
    -------
    dict
        Mapping from an int to a list of performance constraints
        (Constraint objects),
        for which the int is equal to the specified priority.
        Keys are sorted in descending order
        (i.e. highest priority first).
    """
    separation_priority_groups = dict()
    config_sep_priority_dict = config.separation_priority_order
    for perf_con in model_data.separation_model.util.performance_constraints:
        priority = config_sep_priority_dict.get(perf_con.name, 0)
        cons_with_same_priority = separation_priority_groups.setdefault(priority, [])
        cons_with_same_priority.append(perf_con)
    return {priority: perf_cons for priority, perf_cons in sorted(separation_priority_groups.items(), reverse=True)}