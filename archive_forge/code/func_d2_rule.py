from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def d2_rule(disjunct, flag):
    if not flag:
        disjunct.c = Constraint(expr=m.a >= 30)
    else:
        disjunct.c = Constraint(expr=m.a == 100)