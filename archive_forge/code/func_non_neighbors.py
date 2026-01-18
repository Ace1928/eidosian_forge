from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def non_neighbors(graph, node):
    """Returns the non-neighbors of the node in the graph.

    Parameters
    ----------
    graph : NetworkX graph
        Graph to find neighbors.

    node : node
        The node whose neighbors will be returned.

    Returns
    -------
    non_neighbors : iterator
        Iterator of nodes in the graph that are not neighbors of the node.
    """
    nbors = set(neighbors(graph, node)) | {node}
    return (nnode for nnode in graph if nnode not in nbors)