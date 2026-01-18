from pyomo.core import (
from pyomo.core.expr import sqrt
from pyomo.gdp import Disjunct, Disjunction
import pyomo.network as ntwk
def makeExpandedNetworkDisjunction(minimize=True):
    m = makeNetworkDisjunction(minimize)
    TransformationFactory('network.expand_arcs').apply_to(m)
    return m