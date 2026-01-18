from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermDisj_boxes():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 5))
    m.y = Var(bounds=(0, 5))

    def d_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c1 = Constraint(expr=inequality(1, m.x, 2))
            disjunct.c2 = Constraint(expr=inequality(3, m.y, 4))
        else:
            disjunct.c1 = Constraint(expr=inequality(3, m.x, 4))
            disjunct.c2 = Constraint(expr=inequality(1, m.y, 2))
    m.d = Disjunct([0, 1], rule=d_rule)

    def disj_rule(m):
        return [m.d[0], m.d[1]]
    m.disjunction = Disjunction(rule=disj_rule)
    m.obj = Objective(expr=m.x + 2 * m.y)
    return m