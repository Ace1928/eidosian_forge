import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def initialize_unit(self):
    for i in self.flow_out:
        self.flow_out[i].value = value(self.flow_in[i])
    for i in self.expr_var_idx_out:
        self.expr_var_idx_out[i].value = value(self.expr_var_idx_in[i])
    self.expr_var_out.value = value(self.expr_var_in)
    self.temperature_out.value = value(self.temperature_in)
    self.pressure_out.value = value(self.pressure_in)