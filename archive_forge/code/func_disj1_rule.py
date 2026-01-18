from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def disj1_rule(disjunct):
    m = disjunct.model()

    def c_rule(d, s):
        return m.a[s] == 0
    disjunct.c = Constraint(m.s, rule=c_rule)