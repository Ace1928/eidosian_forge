import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.core.base.component import ComponentData
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters
from pyomo.contrib.sensitivity_toolbox.sens import (
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_example
from pyomo.opt import SolverFactory
from pyomo.common.dependencies import (
from pyomo.common.dependencies import scipy_available
def make_model_with_inequalities():
    """
    Creates a modified version of the model used in the "parameter.py"
    example, now with simple (one-sided) inequalities.
    """
    m = ConcreteModel()
    m.x = Var([1, 2, 3], initialize={1: 0.15, 2: 0.15, 3: 0.0}, domain=NonNegativeReals)
    m.eta = Param([1, 2], initialize={1: 4.5, 2: 1.0}, mutable=True)
    m.const = Constraint([1, 2], rule={1: 6 * m.x[1] + 3 * m.x[2] + 2 * m.x[3] >= m.eta[1], 2: m.eta[2] * m.x[1] + m.x[2] - m.x[3] - 1 <= 0})
    m.cost = Objective(expr=m.x[1] ** 2 + m.x[2] ** 2 + m.x[3] ** 2)
    return m