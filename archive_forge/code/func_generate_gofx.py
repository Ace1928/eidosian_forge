from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def generate_gofx(self, G, tears):
    edge_list = self.idx_to_edge(G)
    gofx = []
    for tear in tears:
        arc = G.edges[edge_list[tear]]['arc']
        src = arc.src
        sf = arc.expanded_block.component('splitfrac')
        for name, index, mem in src.iter_vars(names=True):
            if src.is_extensive(name) and sf is not None:
                gofx.append(value(mem * sf))
            else:
                gofx.append(value(mem))
    gofx = numpy.array(gofx)
    return gofx