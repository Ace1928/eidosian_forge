from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def pass_edges(self, G, edges):
    """Call pass values for a list of edge indexes"""
    fixed_outputs = ComponentSet()
    edge_list = self.idx_to_edge(G)
    for ei in edges:
        arc = G.edges[edge_list[ei]]['arc']
        for var in arc.src.iter_vars(expr_vars=True, fixed=False):
            fixed_outputs.add(var)
            var.fix()
        self.pass_values(arc, self.fixed_inputs())
        for var in fixed_outputs:
            var.free()
        fixed_outputs.clear()