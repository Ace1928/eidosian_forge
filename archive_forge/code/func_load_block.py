import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.network import Arc, Port
from pyomo.core.expr.visitor import identify_variables
from pyomo.common.collections.component_set import ComponentSet
def load_block(b):
    b.p_in = Var(b.model().time)
    b.inlet = Port(initialize={'p': (b.p_in, Port.Extensive, {'include_splitfrac': False})})