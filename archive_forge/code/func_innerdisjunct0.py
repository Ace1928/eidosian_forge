from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
@disjunct.Disjunct()
def innerdisjunct0(disjunct):
    disjunct.c = Constraint(expr=m.x <= 2)