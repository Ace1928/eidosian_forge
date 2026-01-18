from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeBetweenStepsPaperExample_DeclareVarOnDisjunct():
    """Exactly the same model as above, but declaring the Disjuncts explicitly
    and declaring the variables on one of them.
    """
    m = ConcreteModel()
    m.I = RangeSet(1, 4)
    m.disj1 = Disjunct()
    m.disj1.x = Var(m.I, bounds=(-2, 6))
    m.disj1.c = Constraint(expr=sum((m.disj1.x[i] ** 2 for i in m.I)) <= 1)
    m.disj2 = Disjunct()
    m.disj2.c = Constraint(expr=sum(((3 - m.disj1.x[i]) ** 2 for i in m.I)) <= 1)
    m.disjunction = Disjunction(expr=[m.disj1, m.disj2])
    m.obj = Objective(expr=m.disj1.x[2] - m.disj1.x[1], sense=maximize)
    return m