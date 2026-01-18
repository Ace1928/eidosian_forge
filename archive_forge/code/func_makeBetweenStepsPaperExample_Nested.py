from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeBetweenStepsPaperExample_Nested():
    """Mathematically, this is really dumb, but I am nesting this model on
    itself because it makes writing tests simpler (I can recycle.)"""
    m = makeBetweenStepsPaperExample_DeclareVarOnDisjunct()
    m.disj2.disjunction = Disjunction(expr=[[sum((m.disj1.x[i] ** 2 for i in m.I)) <= 1], [sum(((3 - m.disj1.x[i]) ** 2 for i in m.I)) <= 1]])
    return m