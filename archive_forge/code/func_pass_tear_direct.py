from pyomo.network import Port, Arc
from pyomo.network.foqus_graph import FOQUSGraph
from pyomo.core import (
from pyomo.common.collections import ComponentSet, ComponentMap, Bunch
from pyomo.core.expr import identify_variables
from pyomo.repn import generate_standard_repn
import logging, time
from pyomo.common.dependencies import (
imports_available = networkx_available & numpy_available
def pass_tear_direct(self, G, tears):
    """Pass values across all tears in the given tear set"""
    fixed_outputs = ComponentSet()
    edge_list = self.idx_to_edge(G)
    for tear in tears:
        arc = G.edges[edge_list[tear]]['arc']
        for var in arc.src.iter_vars(expr_vars=True, fixed=False):
            fixed_outputs.add(var)
            var.fix()
        self.pass_values(arc, fixed_inputs=self.fixed_inputs())
        for var in fixed_outputs:
            var.free()
        fixed_outputs.clear()