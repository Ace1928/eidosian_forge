from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def create_nested_model(self):
    """
        -100 <= x <= 102
        [-10 <= x <= 11, [x <= 3] v [x >= -7]] v [x == 0]
        """
    m = self.create_nested_structure()
    m.x = Var(bounds=(-100, 102))
    m.outer_d1.c = Constraint(expr=(-10, m.x, 11))
    m.outer_d1.inner_d1.c = Constraint(expr=m.x <= 3)
    m.outer_d1.inner_d2.c = Constraint(expr=m.x >= -7)
    m.outer_d2.c = Constraint(expr=m.x == 0)
    return m