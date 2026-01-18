from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermIndexedDisjunction_BoundedVars():
    """Two-term indexed disjunction.
    Adds nothing to above--exists for historic reasons"""
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2, 3])
    m.a = Var(m.s, bounds=(-100, 100))

    def disjunct_rule(d, s, flag):
        m = d.model()
        if flag:
            d.c = Constraint(expr=m.a[s] >= 6)
        else:
            d.c = Constraint(expr=m.a[s] <= 3)
    m.disjunct = Disjunct(m.s, [0, 1], rule=disjunct_rule)

    def disjunction_rule(m, s):
        return [m.disjunct[s, flag] for flag in [0, 1]]
    m.disjunction = Disjunction(m.s, rule=disjunction_rule)
    return m