import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def number_weakly_connected_components(G):
    """Returns the number of weakly connected components in G.

    Parameters
    ----------
    G : NetworkX graph
        A directed graph.

    Returns
    -------
    n : integer
        Number of weakly connected components

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (2, 1), (3, 4)])
    >>> nx.number_weakly_connected_components(G)
    2

    See Also
    --------
    weakly_connected_components
    number_connected_components
    number_strongly_connected_components

    Notes
    -----
    For directed graphs only.

    """
    return sum((1 for wcc in weakly_connected_components(G)))