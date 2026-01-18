from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def oneVarDisj_2pts():
    m = ConcreteModel()
    m.x = Var(bounds=(0, 10))
    m.disj1 = Disjunct()
    m.disj1.xTrue = Constraint(expr=m.x == 1)
    m.disj2 = Disjunct()
    m.disj2.xFalse = Constraint(expr=m.x == 0)
    m.disjunction = Disjunction(expr=[m.disj1, m.disj2])
    m.obj = Objective(expr=m.x)
    return m