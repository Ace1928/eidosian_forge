from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def silly_tabular_data_rule(m, i):
    if i == 1:
        return {1: 1, 3: 9, 6: 36, 10: 100}
    if i == 2:
        return {1: 0, 3: log(3), 6: log(6), 10: log(10)}