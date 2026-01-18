import networkx as nx
from networkx.utils.decorators import not_implemented_for
from ...utils import arbitrary_element
@nx._dispatch
def number_connected_components(G):
    """Returns the number of connected components.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Returns
    -------
    n : integer
       Number of connected components

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])
    >>> nx.number_connected_components(G)
    3

    See Also
    --------
    connected_components
    number_weakly_connected_components
    number_strongly_connected_components

    Notes
    -----
    For undirected graphs only.

    """
    return sum((1 for cc in connected_components(G)))