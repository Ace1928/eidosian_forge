from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def grossmann_twoDisj():
    m = grossmann_oneDisj()
    m.disjunct3 = Disjunct()
    m.disjunct3.constraintx = Constraint(expr=inequality(1, m.x, 2.5))
    m.disjunct3.constrainty = Constraint(expr=inequality(6.5, m.y, 8))
    m.disjunct4 = Disjunct()
    m.disjunct4.constraintx = Constraint(expr=inequality(9, m.x, 11))
    m.disjunct4.constrainty = Constraint(expr=inequality(2, m.y, 3.5))
    m.disjunction2 = Disjunction(expr=[m.disjunct3, m.disjunct4])
    return m