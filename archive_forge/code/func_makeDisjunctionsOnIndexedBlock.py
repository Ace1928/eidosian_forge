from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeDisjunctionsOnIndexedBlock():
    """Two disjunctions (one indexed and one not), each on a separate
    BlockData of an IndexedBlock of length 2
    """
    m = ConcreteModel()
    m.s = Set(initialize=[1, 2])
    m.a = Var(m.s, bounds=(0, 70))

    @m.Disjunct(m.s, [0, 1])
    def disjunct1(disjunct, s, flag):
        m = disjunct.model()
        if not flag:
            disjunct.c = Constraint(expr=m.a[s] == 0)
        else:
            disjunct.c = Constraint(expr=m.a[s] >= 7)

    def disjunction1_rule(m, s):
        return [m.disjunct1[s, flag] for flag in [0, 1]]
    m.disjunction1 = Disjunction(m.s, rule=disjunction1_rule)
    m.b = Block([0, 1])
    m.b[0].x = Var(bounds=(-2, 2))

    def disjunct2_rule(disjunct, flag):
        if not flag:
            disjunct.c = Constraint(expr=m.b[0].x <= 0)
        else:
            disjunct.c = Constraint(expr=m.b[0].x >= 0)
    m.b[0].disjunct = Disjunct([0, 1], rule=disjunct2_rule)

    def disjunction(b, i):
        return [b.disjunct[0], b.disjunct[1]]
    m.b[0].disjunction = Disjunction([0], rule=disjunction)
    m.b[1].y = Var(bounds=(-3, 3))
    m.b[1].disjunct0 = Disjunct()
    m.b[1].disjunct0.c = Constraint(expr=m.b[1].y <= 0)
    m.b[1].disjunct1 = Disjunct()
    m.b[1].disjunct1.c = Constraint(expr=m.b[1].y >= 0)
    m.b[1].disjunction = Disjunction(expr=[m.b[1].disjunct0, m.b[1].disjunct1])
    return m