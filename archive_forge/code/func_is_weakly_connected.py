import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def is_weakly_connected(G):
    """Test directed graph for weak connectivity.

    A directed graph is weakly connected if and only if the graph
    is connected when the direction of the edge between nodes is ignored.

    Note that if a graph is strongly connected (i.e. the graph is connected
    even when we account for directionality), it is by definition weakly
    connected as well.

    Parameters
    ----------
    G : NetworkX Graph
        A directed graph.

    Returns
    -------
    connected : bool
        True if the graph is weakly connected, False otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (2, 1)])
    >>> G.add_node(3)
    >>> nx.is_weakly_connected(G)  # node 3 is not connected to the graph
    False
    >>> G.add_edge(2, 3)
    >>> nx.is_weakly_connected(G)
    True

    See Also
    --------
    is_strongly_connected
    is_semiconnected
    is_connected
    is_biconnected
    weakly_connected_components

    Notes
    -----
    For directed graphs only.

    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('Connectivity is undefined for the null graph.')
    return len(next(weakly_connected_components(G))) == len(G)