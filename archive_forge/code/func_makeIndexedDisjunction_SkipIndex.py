from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeIndexedDisjunction_SkipIndex():
    """Two-term indexed disjunction where one of the two indices is skipped"""
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))

    @m.Disjunct([0, 1])
    def disjuncts(d, i):
        m = d.model()
        d.cons = Constraint(expr=m.x == i)

    @m.Disjunction([0, 1])
    def disjunctions(m, i):
        if i == 0:
            return Disjunction.Skip
        return [m.disjuncts[i], m.disjuncts[0]]
    return m