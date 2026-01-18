from contextlib import contextmanager
import logging
from math import fabs
import sys
from pyomo.common import timing
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.core import (
from pyomo.core.expr.numvalue import native_types
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.opt import SolverFactory
def fix_discrete_var(var, val, config):
    """Fixes the discrete variable var to val, rounding to the nearest integer
    or not, depending on if rounding is specified in config and what the integer
    tolerance is."""
    if val is None:
        return
    if var.is_continuous():
        var.set_value(val, skip_validation=True)
    elif fabs(val - round(val)) > config.integer_tolerance:
        raise ValueError("Integer variable '%s' cannot be fixed to value %s because it is not within the specified integer tolerance of %s." % (var.name, val, config.integer_tolerance))
    elif config.round_discrete_vars:
        var.fix(int(round(val)))
    else:
        var.fix(val, skip_validation=True)