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
def solve_separation_problem(model_data, config):
    """
    Solve PyROS separation problems.

    Parameters
    ----------
    model_data : SeparationProblemData
        Separation problem data.
    config : ConfigDict
        PyROS solver settings.

    Returns
    -------
    pyros.solve_data.SeparationResults
        Separation problem solve results.
    """
    run_local = not config.bypass_local_separation
    run_global = config.bypass_local_separation
    uncertainty_set_is_discrete = config.uncertainty_set.geometry == Geometry.DISCRETE_SCENARIOS
    if run_local:
        local_separation_loop_results = perform_separation_loop(model_data=model_data, config=config, solve_globally=False)
        run_global = not (local_separation_loop_results.found_violation or uncertainty_set_is_discrete or local_separation_loop_results.subsolver_error or local_separation_loop_results.time_out or config.bypass_global_separation)
    else:
        local_separation_loop_results = None
    if run_global:
        global_separation_loop_results = perform_separation_loop(model_data=model_data, config=config, solve_globally=True)
    else:
        global_separation_loop_results = None
    return SeparationResults(local_separation_loop_results=local_separation_loop_results, global_separation_loop_results=global_separation_loop_results)