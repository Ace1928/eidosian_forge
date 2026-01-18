import copy, logging
from pyomo.common.dependencies import numpy
def sub_graph_edges(self, G, nodes):
    """
        This function returns a list of edge indexes that are
        included in a subgraph given by a list of nodes.

        Returns
        -------
            edges
                List of edge indexes in the subgraph
            inEdges
                List of edge indexes starting outside the subgraph
                and ending inside
            outEdges
                List of edge indexes starting inside the subgraph
                and ending outside
        """
    e = []
    ie = []
    oe = []
    edge_list = self.idx_to_edge(G)
    for i in range(G.number_of_edges()):
        src, dest, _ = edge_list[i]
        if src in nodes:
            if dest in nodes:
                e.append(i)
            else:
                oe.append(i)
        elif dest in nodes:
            ie.append(i)
    return (e, ie, oe)