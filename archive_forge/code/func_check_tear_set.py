import copy, logging
from pyomo.common.dependencies import numpy
def check_tear_set(self, G, tset):
    """
        Check whether the specified tear streams are sufficient.
        If the graph minus the tear edges is not a tree then the
        tear set is not sufficient to solve the graph.
        """
    sccNodes, _, _, _ = self.scc_collect(G, excludeEdges=tset)
    for nodes in sccNodes:
        if len(nodes) > 1:
            return False
    return True