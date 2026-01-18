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
def higher_order_decision_rule_efficiency(model_data, config):
    """
    Enforce DR coefficient variable efficiencies for
    master problem-like formulation.

    Parameters
    ----------
    model_data : MasterProblemData
        Master problem data.
    config : ConfigDict
        PyROS solver options.

    Note
    ----
    The DR coefficient variable efficiencies consist of
    setting the degree of the DR polynomial expressions
    by fixing the appropriate variables to 0. The degree
    to be set depends on the iteration number;
    see ``get_master_dr_degree``.
    """
    order_to_enforce = get_master_dr_degree(model_data, config)
    enforce_dr_degree(blk=model_data.master_model.scenarios[0, 0], config=config, degree=order_to_enforce)