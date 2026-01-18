from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeNonQuadraticNonlinearGDP():
    """We use this in testing between steps--Needed non-quadratic and not
    additively separable constraint expressions on a Disjunct."""
    m = ConcreteModel()
    m.I = RangeSet(1, 4)
    m.I1 = RangeSet(1, 2)
    m.I2 = RangeSet(3, 4)
    m.x = Var(m.I, bounds=(-2, 6))
    m.disjunction = Disjunction(expr=[[sum((m.x[i] ** 4 for i in m.I1)) ** (1 / 4) + sum((m.x[i] ** 4 for i in m.I2)) ** (1 / 4) <= 1], [sum(((3 - m.x[i]) ** 4 for i in m.I1)) ** (1 / 4) + sum(((3 - m.x[i]) ** 4 for i in m.I2)) ** (1 / 4) <= 1]])
    m.obj = Objective(expr=m.x[2] - m.x[1], sense=maximize)
    return m