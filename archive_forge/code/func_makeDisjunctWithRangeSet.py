from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeDisjunctWithRangeSet():
    """Two-term SimpleDisjunction where one of the disjuncts contains a
    RangeSet"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.d1 = Disjunct()
    m.d1.s = RangeSet(1)
    m.d1.c = Constraint(rule=lambda _: m.x == 1)
    m.d2 = Disjunct()
    m.disj = Disjunction(expr=[m.d1, m.d2])
    return m