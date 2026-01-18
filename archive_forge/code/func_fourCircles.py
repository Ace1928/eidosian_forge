from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def fourCircles():
    m = twoDisj_twoCircles_easy()
    m.upper_circle2 = Disjunct()
    m.upper_circle2.cons = Constraint(expr=(m.x - 2) ** 2 + (m.y - 7) ** 2 <= 1)
    m.lower_circle2 = Disjunct()
    m.lower_circle2.cons = Constraint(expr=(m.x - 5) ** 2 + (m.y - 3) ** 2 <= 2)
    m.disjunction2 = Disjunction(expr=[m.upper_circle2, m.lower_circle2])
    return m