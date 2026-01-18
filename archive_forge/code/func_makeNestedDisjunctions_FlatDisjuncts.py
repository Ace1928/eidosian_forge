from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeNestedDisjunctions_FlatDisjuncts():
    """Two-term SimpleDisjunction where one of the disjuncts contains a nested
    SimpleDisjunction, the disjuncts of which are declared on the model"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 2))
    m.obj = Objective(expr=m.x)
    m.d1 = Disjunct()
    m.d1.c = Constraint(expr=m.x >= 1)
    m.d2 = Disjunct()
    m.d2.c = Constraint(expr=m.x >= 1.1)
    m.d3 = Disjunct()
    m.d3.c = Constraint(expr=m.x >= 1.2)
    m.d4 = Disjunct()
    m.d4.c = Constraint(expr=m.x >= 1.3)
    m.disj = Disjunction(expr=[m.d1, m.d2])
    m.d1.disj = Disjunction(expr=[m.d3, m.d4])
    return m