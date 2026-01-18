from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
@m.Disjunct(m.s, [0, 1])
def disjunct1(disjunct, s, flag):
    m = disjunct.model()
    if not flag:
        disjunct.c = Constraint(expr=m.a[s] == 0)
    else:
        disjunct.c = Constraint(expr=m.a[s] >= 7)