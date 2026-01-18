import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def is_strongly_connected(G):
    """Test directed graph for strong connectivity.

    A directed graph is strongly connected if and only if every vertex in
    the graph is reachable from every other vertex.

    Parameters
    ----------
    G : NetworkX Graph
       A directed graph.

    Returns
    -------
    connected : bool
      True if the graph is strongly connected, False otherwise.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 2)])
    >>> nx.is_strongly_connected(G)
    True
    >>> G.remove_edge(2, 3)
    >>> nx.is_strongly_connected(G)
    False

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    See Also
    --------
    is_weakly_connected
    is_semiconnected
    is_connected
    is_biconnected
    strongly_connected_components

    Notes
    -----
    For directed graphs only.
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Connectivity is undefined for the null graph.')
    return len(next(strongly_connected_components(G))) == len(G)