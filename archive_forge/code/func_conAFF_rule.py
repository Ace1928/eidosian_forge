import logging
import math
import itertools
import operator
import types
import enum
from pyomo.common.log import is_debug_set
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import value
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.block import Block, _BlockData
from pyomo.core.base.component import ModelComponentFactory
from pyomo.core.base.constraint import Constraint, ConstraintList
from pyomo.core.base.sos import SOSConstraint
from pyomo.core.base.var import Var, _VarData, IndexedVar
from pyomo.core.base.set_types import PositiveReals, NonNegativeReals, Binary
from pyomo.core.base.util import flatten_tuple
def conAFF_rule(model, i):
    rhs = 1.0
    if i not in OPT_M['LB']:
        rhs *= 0.0
    else:
        rhs *= OPT_M['LB'][i] * (1 - bigm_y[i])
    return y_var - y_pts[i - 1] - (y_pts[i] - y_pts[i - 1]) / (x_pts[i] - x_pts[i - 1]) * (x_var - x_pts[i - 1]) >= rhs