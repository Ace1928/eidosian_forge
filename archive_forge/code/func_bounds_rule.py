from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def bounds_rule(m, s):
    return (m.lbs[s], m.ubs[s])