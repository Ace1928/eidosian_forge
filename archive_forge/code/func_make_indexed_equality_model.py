from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def make_indexed_equality_model():
    """
    min  x_1 + x_2
    s.t. [x_1 = 1] v [x_1 = 2]
         [x_2 = 1] v [x_2 = 2]
    """

    def disj_rule(m, t):
        return [[m.x[t] == 1], [m.x[t] == 2]]
    m = ConcreteModel()
    m.T = RangeSet(2)
    m.x = Var(m.T, within=NonNegativeReals, bounds=(0, 5))
    m.d = Disjunction(m.T, rule=disj_rule)
    m.obj = Objective(expr=m.x[1] + m.x[2], sense=minimize)
    return m