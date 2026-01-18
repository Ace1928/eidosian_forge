import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
@m.splitter.Port()
def outlet_side_2(b):
    return dict(flow=b.flow_out_side_2, temperature=b.temperature_out_side_2, pressure=b.pressure_out_side_2, expr_idx=b.expr_idx_out_side_2, expr=b.expr_out_side_2)