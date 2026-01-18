from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def tear_set_arcs(self, G, method='mip', **kwds):
    """
        Call the specified tear selection method and return a list
        of arcs representing the selected tear edges.

        The kwds will be passed to the method.
        """
    if method == 'mip':
        tset = self.select_tear_mip(G, **kwds)
    elif method == 'heuristic':
        tset = self.select_tear_heuristic(G, **kwds)[0][0]
    else:
        raise ValueError("Invalid method '%s'" % (method,))
    return self.indexes_to_arcs(G, tset)