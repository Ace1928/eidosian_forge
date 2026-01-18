import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def build_in_out(b):
    b.flow_in = Var(m.comps)
    b.mass_in = Var()
    b.temperature_in = Var()
    b.pressure_in = Var()
    b.expr_var_idx_in = Var(m.comps)

    @b.Expression(m.comps)
    def expr_idx_in(b, i):
        return -b.expr_var_idx_in[i]
    b.expr_var_in = Var()
    b.expr_in = -b.expr_var_in
    b.flow_out = Var(m.comps)
    b.mass_out = Var()
    b.temperature_out = Var()
    b.pressure_out = Var()
    b.expr_var_idx_out = Var(m.comps)

    @b.Expression(m.comps)
    def expr_idx_out(b, i):
        return -b.expr_var_idx_out[i]
    b.expr_var_out = Var()
    b.expr_out = -b.expr_var_out
    b.inlet = Port(rule=inlet)
    b.outlet = Port(rule=outlet)
    b.initialize = MethodType(initialize, b)