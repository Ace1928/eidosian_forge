from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermMultiIndexedDisjunction():
    """Two-term indexed disjunction with tuple indices"""
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.t = Set(initialize=['A', 'B'])
    m.a = Var(m.s, m.t, bounds=(2, 7))

    def d_rule(disjunct, flag, s, t):
        m = disjunct.model()
        if flag:
            disjunct.c = Constraint(expr=m.a[s, t] == 0)
        else:
            disjunct.c = Constraint(expr=m.a[s, t] >= 5)
    m.disjunct = Disjunct([0, 1], m.s, m.t, rule=d_rule)

    def disj_rule(m, s, t):
        return [m.disjunct[0, s, t], m.disjunct[1, s, t]]
    m.disjunction = Disjunction(m.s, m.t, rule=disj_rule)
    return m