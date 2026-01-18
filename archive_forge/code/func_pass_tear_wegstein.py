from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def pass_tear_wegstein(self, G, tears, x):
    """
        Set the destination value of all tear edges to
        the corresponding value in the numpy array x.
        """
    fixed_inputs = self.fixed_inputs()
    edge_list = self.idx_to_edge(G)
    i = 0
    for tear in tears:
        arc = G.edges[edge_list[tear]]['arc']
        src, dest = (arc.src, arc.dest)
        dest_unit = dest.parent_block()
        if dest_unit not in fixed_inputs:
            fixed_inputs[dest_unit] = ComponentSet()
        for name, index, mem in src.iter_vars(names=True):
            peer = self.source_dest_peer(arc, name, index)
            self.pass_single_value(dest, name, peer, x[i], fixed_inputs[dest_unit])
            i += 1