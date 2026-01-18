from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeThreeTermDisj_IndexedConstraints():
    """Three-term disjunction with indexed constraints on the disjuncts"""
    m = ConcreteModel()
    m.I = [1, 2, 3]
    m.x = Var(m.I, bounds=(0, 10))

    def c_rule(b, i):
        m = b.model()
        return m.x[i] >= i

    def d_rule(d, j):
        m = d.model()
        d.c = Constraint(m.I[:j], rule=c_rule)
    m.d = Disjunct(m.I, rule=d_rule)
    m.disjunction = Disjunction(expr=[m.d[i] for i in m.I])
    return m