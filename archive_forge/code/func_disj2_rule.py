from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def disj2_rule(disjunct):
    m = disjunct.model()

    def c_rule(d, s):
        return m.a[s] <= 3
    disjunct.c = Constraint(m.s, rule=c_rule)