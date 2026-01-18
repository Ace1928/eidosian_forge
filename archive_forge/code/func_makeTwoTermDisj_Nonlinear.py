from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermDisj_Nonlinear():
    """Single two-term disjunction which has all of ==, <=, and >= and
    one nonlinear constraint.
    """
    m = ConcreteModel()
    m.w = Var(bounds=(2, 7))
    m.x = Var(bounds=(1, 8))
    m.y = Var(bounds=(-10, -3))

    def d_rule(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c1 = Constraint(expr=m.x >= 2)
            disjunct.c2 = Constraint(expr=m.w == 3)
            disjunct.c3 = Constraint(expr=(1, m.x, 3))
        else:
            disjunct.c = Constraint(expr=m.x + m.y ** 2 <= 14)
    m.d = Disjunct([0, 1], rule=d_rule)
    m.disjunction = Disjunction(expr=[m.d[0], m.d[1]])
    return m