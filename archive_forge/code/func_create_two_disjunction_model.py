from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import _parent_disjunct
from pyomo.common.log import LoggingIntercept
def create_two_disjunction_model(self):
    m = self.create_nested_model()
    m.y = Var()
    m.d1 = Disjunct()
    m.d2 = Disjunct()
    m.d3 = Disjunct()
    m.disjunction = Disjunction(expr=[m.d1, m.d2, m.d3])
    m.d1.c = Constraint(expr=m.y == 7.8)
    m.d1.c_x = Constraint(expr=m.x <= 27)
    m.d2.c = Constraint(expr=m.y == 8.9)
    m.d2.c_x = Constraint(expr=m.x >= 34)
    m.d3.c = Constraint(expr=m.y <= 45.7)
    return m