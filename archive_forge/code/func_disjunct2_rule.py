from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def disjunct2_rule(disjunct, flag):
    if not flag:
        disjunct.c = Constraint(expr=m.b[0].x <= 0)
    else:
        disjunct.c = Constraint(expr=m.b[0].x >= 0)