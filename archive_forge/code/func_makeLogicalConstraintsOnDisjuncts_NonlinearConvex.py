from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeLogicalConstraintsOnDisjuncts_NonlinearConvex():
    m = ConcreteModel()
    m.s = RangeSet(4)
    m.ds = RangeSet(2)
    m.d = Disjunct(m.s)
    m.djn = Disjunction(m.ds)
    m.djn[1] = [m.d[1], m.d[2]]
    m.djn[2] = [m.d[3], m.d[4]]
    m.x = Var(bounds=(-5, 10))
    m.y = Var(bounds=(-5, 10))
    m.Y = BooleanVar([1, 2])
    m.d[1].c = Constraint(expr=m.x ** 2 + m.y ** 2 <= 2)
    m.d[1].logical = LogicalConstraint(expr=~m.Y[1])
    m.d[2].c1 = Constraint(expr=m.x >= -3)
    m.d[2].c2 = Constraint(expr=m.x ** 2 <= 16)
    m.d[2].logical = LogicalConstraint(expr=m.Y[1].land(m.Y[2]))
    m.d[3].c = Constraint(expr=m.x >= 4)
    m.d[4].logical = LogicalConstraint(expr=exactly(1, m.Y[1]))
    m.d[4].logical2 = LogicalConstraint(expr=~m.Y[2])
    m.d[4].c = Constraint(expr=m.x == 3)
    m.o = Objective(expr=m.x)
    return m