import networkx as nx
from networkx.utils.decorators import not_implemented_for
from ...utils import arbitrary_element
@not_implemented_for('directed')
@nx._dispatch
def node_connected_component(G, n):
    """Returns the set of nodes in the component of graph containing node n.

    Parameters
    ----------
    G : NetworkX Graph
       An undirected graph.

    n : node label
       A node in G

    Returns
    -------
    comp : set
       A set of nodes in the component of G containing node n.

    Raises
    ------
    NetworkXNotImplemented
        If G is directed.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (5, 6), (3, 4)])
    >>> nx.node_connected_component(G, 0)  # nodes of component that contains node 0
    {0, 1, 2}

    See Also
    --------
    connected_components

    Notes
    -----
    For undirected graphs only.

    """
    return _plain_bfs(G, n)