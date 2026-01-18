from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def node_to_idx(self, G):
    """Returns a mapping from nodes to indexes for a graph"""

    def fcn(G):
        res = dict()
        i = -1
        for node in G.nodes:
            i += 1
            res[node] = i
        return res
    return self.cacher('node_to_idx', fcn, G)