import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import prod
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import (
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.core.expr.numeric_expr import SumExpression
from pyomo.environ import SolverFactory
import itertools as it
import timeit
from contextlib import contextmanager
import logging
import math
from pyomo.common.timing import HierarchicalTimer
from pyomo.common.log import Preformatted
def selective_clone(block, first_stage_vars):
    """
    Clone everything in a base_model except for the first-stage variables
    :param block: the block of the model to be clones
    :param first_stage_vars: the variables which should not be cloned
    :return:
    """
    memo = {'__block_scope__': {id(block): True, id(None): False}}
    for v in first_stage_vars:
        memo[id(v)] = v
    new_block = copy.deepcopy(block, memo)
    new_block._parent = None
    return new_block