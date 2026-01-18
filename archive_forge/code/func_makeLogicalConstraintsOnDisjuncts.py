from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeLogicalConstraintsOnDisjuncts():
    m = ConcreteModel()
    m.s = RangeSet(4)
    m.ds = RangeSet(2)
    m.d = Disjunct(m.s)
    m.djn = Disjunction(m.ds)
    m.djn[1] = [m.d[1], m.d[2]]
    m.djn[2] = [m.d[3], m.d[4]]
    m.x = Var(bounds=(-2, 10))
    m.Y = BooleanVar([1, 2])
    m.d[1].c = Constraint(expr=m.x >= 2)
    m.d[1].logical = LogicalConstraint(expr=~m.Y[1])
    m.d[2].c = Constraint(expr=m.x >= 3)
    m.d[3].c = Constraint(expr=m.x >= 8)
    m.d[4].logical = LogicalConstraint(expr=m.Y[1].equivalent_to(m.Y[2]))
    m.d[4].c = Constraint(expr=m.x == 2.5)
    m.o = Objective(expr=m.x)
    m.p = LogicalConstraint(expr=m.d[1].indicator_var.implies(m.d[4].indicator_var))
    m.bwahaha = LogicalConstraint(expr=m.Y[1].xor(m.Y[2]))
    return m