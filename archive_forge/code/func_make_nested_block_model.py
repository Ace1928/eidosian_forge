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
def make_nested_block_model(self):
    """For the next two tests: Has BooleanVar on model, but
        LogicalConstraints on a Block and a Block nested on that Block."""
    m = ConcreteModel()
    m.b = Block()
    m.Y = BooleanVar([1, 2])
    m.b.logical = LogicalConstraint(expr=~m.Y[1])
    m.b.b = Block()
    m.b.b.logical = LogicalConstraint(expr=m.Y[1].xor(m.Y[2]))
    return m