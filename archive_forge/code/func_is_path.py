from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def is_path(G, path):
    """Returns whether or not the specified path exists.

    For it to return True, every node on the path must exist and
    each consecutive pair must be connected via one or more edges.

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    path : list
        A list of nodes which defines the path to traverse

    Returns
    -------
    bool
        True if `path` is a valid path in `G`

    """
    return all((node in G and nbr in G[node] for node, nbr in nx.utils.pairwise(path)))