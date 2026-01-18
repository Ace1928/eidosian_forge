from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermIndexedDisjunction():
    """Two-term indexed disjunction"""
    m = ConcreteModel()
    m.A = Set(initialize=[1, 2, 3])
    m.B = Set(initialize=['a', 'b'])
    m.x = Var(m.A, bounds=(-10, 10))

    def disjunct_rule(d, i, k):
        m = d.model()
        if k == 'a':
            d.cons_a = Constraint(expr=m.x[i] >= 5)
        if k == 'b':
            d.cons_b = Constraint(expr=m.x[i] <= 0)
    m.disjunct = Disjunct(m.A, m.B, rule=disjunct_rule)

    def disj_rule(m, i):
        return [m.disjunct[i, k] for k in m.B]
    m.disjunction = Disjunction(m.A, rule=disj_rule)
    return m