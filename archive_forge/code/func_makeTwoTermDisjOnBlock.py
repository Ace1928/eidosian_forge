from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeTwoTermDisjOnBlock():
    """Two-term SimpleDisjunction on a block"""
    m = ConcreteModel()
    m.b = Block()
    m.a = Var(bounds=(0, 5))

    @m.b.Disjunct([0, 1])
    def disjunct(disjunct, flag):
        m = disjunct.model()
        if flag:
            disjunct.c = Constraint(expr=m.a <= 3)
        else:
            disjunct.c = Constraint(expr=m.a == 0)

    @m.b.Disjunction()
    def disjunction(m):
        return [m.disjunct[0], m.disjunct[1]]
    return m