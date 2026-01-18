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
def get_vars_from_component(block, ctype):
    """Determine all variables used in active components within a block.

    Parameters
    ----------
    block: Block
        The block to search for components.  This is a recursive
        generator and will descend into any active sub-Blocks as well.
    ctype:  class
        The component type (typically either :py:class:`Constraint` or
        :py:class:`Objective` to search for).

    """
    return get_vars_from_components(block, ctype, active=True, descend_into=True)