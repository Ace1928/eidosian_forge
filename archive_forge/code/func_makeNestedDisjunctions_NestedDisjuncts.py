from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeNestedDisjunctions_NestedDisjuncts():
    """Same as makeNestedDisjunctions_FlatDisjuncts except that the disjuncts
    of the nested disjunction are declared on the parent disjunct."""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 2))
    m.obj = Objective(expr=m.x)
    m.d1 = Disjunct()
    m.d1.c = Constraint(expr=m.x >= 1)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x >= 1.1)
    m.d1.d3 = Disjunct()
    m.d1.d3.c = Constraint(expr=m.x >= 1.2)
    m.d1.d4 = Disjunct()
    m.d1.d4.c = Constraint(expr=m.x >= 1.3)
    m.disj = Disjunction(expr=[m.d1, m.d2])
    m.d1.disj2 = Disjunction(expr=[m.d1.d3, m.d1.d4])
    return m