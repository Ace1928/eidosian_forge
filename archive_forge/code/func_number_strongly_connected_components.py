import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def number_strongly_connected_components(G):
    """Returns number of strongly connected components in graph.

    Parameters
    ----------
    G : NetworkX graph
       A directed graph.

    Returns
    -------
    n : integer
       Number of strongly connected components

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    >>> G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (3, 4), (5, 6), (6, 3), (6, 7)])
    >>> nx.number_strongly_connected_components(G)
    3

    See Also
    --------
    strongly_connected_components
    number_connected_components
    number_weakly_connected_components

    Notes
    -----
    For directed graphs only.
    """
    return sum((1 for scc in strongly_connected_components(G)))