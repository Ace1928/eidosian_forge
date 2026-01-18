from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def combine_and_fix(self, port, name, obj, evars, fixed):
    """
        For an extensive port member, combine the values of all
        expanded variables and fix the port member at their sum.
        Assumes that all expanded variables are fixed.
        """
    assert all((evar.is_fixed() for evar in evars))
    total = sum((value(evar) for evar in evars))
    self.pass_single_value(port, name, obj, total, fixed)