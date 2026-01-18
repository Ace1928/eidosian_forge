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
def get_master_dr_degree(model_data, config):
    """
    Determine DR polynomial degree to enforce based on
    the iteration number.

    Currently, the degree is set to:

    - 0 if iteration number is 0
    - min(1, config.decision_rule_order) if iteration number
      otherwise does not exceed number of uncertain parameters
    - min(2, config.decision_rule_order) otherwise.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver options.

    Returns
    -------
    int
        DR order, or polynomial degree, to enforce.
    """
    if model_data.iteration == 0:
        return 0
    elif model_data.iteration <= len(config.uncertain_params):
        return min(1, config.decision_rule_order)
    else:
        return min(2, config.decision_rule_order)