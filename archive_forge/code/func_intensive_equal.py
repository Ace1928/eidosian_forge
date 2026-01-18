import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, networkx_available
from pyomo.environ import (
from pyomo.network import Port, SequentialDecomposition, Arc
from pyomo.gdp.tests.models import makeExpandedNetworkDisjunction
from types import MethodType
import_available = numpy_available and networkx_available
def intensive_equal(self, port, tol=1e-05, **kwds):
    for name in kwds:
        if port.vars[name].is_indexed():
            for i in kwds[name]:
                if abs(value(port.vars[name][i] - kwds[name][i])) > tol:
                    return False
                if abs(value(port.vars[name][i] - kwds[name][i])) > tol:
                    return False
        else:
            if abs(value(port.vars[name] - kwds[name])) > tol:
                return False
            if abs(value(port.vars[name] - kwds[name])) > tol:
                return False
    return True