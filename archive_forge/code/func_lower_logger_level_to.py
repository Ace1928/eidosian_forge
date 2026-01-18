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
@contextmanager
def lower_logger_level_to(logger, level=None, tee=False):
    """Increases logger verbosity by lowering reporting level."""
    if tee:
        level = logging.INFO
        handlers = [h for h in logger.handlers]
        logger.handlers.clear()
        logger.propagate = False
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logger.getEffectiveLevel())
        sh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(sh)
    level_changed = False
    if level is not None and logger.getEffectiveLevel() > level:
        old_logger_level = logger.level
        logger.setLevel(level)
        if tee:
            sh.setLevel(level)
        level_changed = True
    yield
    if tee:
        logger.handlers.clear()
        for h in handlers:
            logger.addHandler(h)
        logger.propagate = True
    if level_changed:
        logger.setLevel(old_logger_level)