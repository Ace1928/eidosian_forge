import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
@m.Disjunct([0, 1])
def disj(disj, _):

    @disj.Disjunct(['A', 'B'])
    def nested(n_disj, _):
        pass
    return disj