import copy, logging
from pyomo.common.dependencies import numpy
def cycle_edge_matrix(self, G):
    """
        Return a cycle-edge incidence matrix, a list of list of nodes in
        each cycle, and a list of list of edge indexes in each cycle.
        """
    cycleNodes, cycleEdges = self.all_cycles(G)
    ceMat = numpy.zeros((len(cycleEdges), G.number_of_edges()), dtype=numpy.dtype(int))
    for i in range(len(cycleEdges)):
        for e in cycleEdges[i]:
            ceMat[i, e] = 1
    return (ceMat, cycleNodes, cycleEdges)