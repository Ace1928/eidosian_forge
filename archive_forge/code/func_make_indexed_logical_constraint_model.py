import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
def make_indexed_logical_constraint_model(self):
    m = _generate_boolean_model(3)
    m.cons = LogicalConstraint([1, 2])
    m.cons[1] = exactly(2, m.Y)
    m.cons[2] = m.Y[1].implies(lor(m.Y[2], m.Y[3]))
    return m