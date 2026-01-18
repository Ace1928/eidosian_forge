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
def revert_solver_max_time_adjustment(solver, original_max_time_setting, custom_setting_present, config):
    """
    Revert solver `options` attribute to its state prior to a
    time limit adjustment performed via
    the routine `adjust_solver_time_settings`.

    Parameters
    ----------
    solver : solver type
        Solver of interest.
    original_max_time_setting : float, list, or None
        Original solver settings. Type depends on the
        solver type.
    custom_setting_present : bool or None
        Was the max time, or other custom solver settings,
        specified prior to the adjustment?
        Can be None if ``config.time_limit`` is None.
    config : ConfigDict
        PyROS solver config.
    """
    if config.time_limit is not None:
        assert isinstance(custom_setting_present, bool)
        if isinstance(solver, type(SolverFactory('gams', solver_io='shell'))):
            options_key = 'add_options'
        elif isinstance(solver, SolverFactory.get_class('baron')):
            options_key = 'MaxTime'
        elif isinstance(solver, SolverFactory.get_class('ipopt')):
            options_key = 'max_cpu_time'
        else:
            options_key = None
        if options_key is not None:
            if custom_setting_present:
                solver.options[options_key] = original_max_time_setting
                if isinstance(solver, type(SolverFactory('gams', solver_io='shell'))):
                    solver.options[options_key].pop()
            else:
                delattr(solver.options, options_key)
                if options_key in solver.options.keys():
                    del solver.options[options_key]