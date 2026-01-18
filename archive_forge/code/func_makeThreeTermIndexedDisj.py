from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeThreeTermIndexedDisj():
    """Three-term indexed disjunction"""
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.a = Var(m.s, bounds=(2, 7))

    def d_rule(disjunct, flag, s):
        m = disjunct.model()
        if flag == 0:
            disjunct.c = Constraint(expr=m.a[s] == 0)
        elif flag == 1:
            disjunct.c = Constraint(expr=m.a[s] >= 5)
        else:
            disjunct.c = Constraint(expr=inequality(2, m.a[s], 4))
    m.disjunct = Disjunct([0, 1, 2], m.s, rule=d_rule)

    def disj_rule(m, s):
        return [m.disjunct[0, s], m.disjunct[1, s], m.disjunct[2, s]]
    m.disjunction = Disjunction(m.s, rule=disj_rule)
    return m