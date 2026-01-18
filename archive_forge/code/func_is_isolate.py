import networkx as nx
@nx._dispatch
def is_isolate(G, n):
    """Determines whether a node is an isolate.

    An *isolate* is a node with no neighbors (that is, with degree
    zero). For directed graphs, this means no in-neighbors and no
    out-neighbors.

    Parameters
    ----------
    G : NetworkX graph

    n : node
        A node in `G`.

    Returns
    -------
    is_isolate : bool
       True if and only if `n` has no neighbors.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge(1, 2)
    >>> G.add_node(3)
    >>> nx.is_isolate(G, 2)
    False
    >>> nx.is_isolate(G, 3)
    True
    """
    return G.degree(n) == 0