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
def process_termination_condition_master_problem(config, results):
    """
    :param config: pyros config
    :param results: solver results object
    :return: tuple (try_backups (True/False)
                  pyros_return_code (default NONE or robust_infeasible or subsolver_error))
    """
    locally_acceptable = [tc.optimal, tc.locallyOptimal, tc.globallyOptimal]
    globally_acceptable = [tc.optimal, tc.globallyOptimal]
    robust_infeasible = [tc.infeasible]
    try_backups = [tc.feasible, tc.maxTimeLimit, tc.maxIterations, tc.maxEvaluations, tc.minStepLength, tc.minFunctionValue, tc.other, tc.solverFailure, tc.internalSolverError, tc.error, tc.unbounded, tc.infeasibleOrUnbounded, tc.invalidProblem, tc.intermediateNonInteger, tc.noSolution, tc.unknown]
    termination_condition = results.solver.termination_condition
    if config.solve_master_globally == False:
        if termination_condition in locally_acceptable:
            return (False, None)
        elif termination_condition in robust_infeasible:
            return (False, pyrosTerminationCondition.robust_infeasible)
        elif termination_condition in try_backups:
            return (True, None)
        else:
            raise NotImplementedError('This solver return termination condition (%s) is currently not supported by PyROS.' % termination_condition)
    elif termination_condition in globally_acceptable:
        return (False, None)
    elif termination_condition in robust_infeasible:
        return (False, pyrosTerminationCondition.robust_infeasible)
    elif termination_condition in try_backups:
        return (True, None)
    else:
        raise NotImplementedError('This solver return termination condition (%s) is currently not supported by PyROS.' % termination_condition)