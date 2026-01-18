import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.exception import AmbiguousSolution
@nx._dispatch(graphs='B', edge_attrs='weight')
def degrees(B, nodes, weight=None):
    """Returns the degrees of the two node sets in the bipartite graph B.

    Parameters
    ----------
    B : NetworkX graph

    nodes: list or container
      Nodes in one node set of the bipartite graph.

    weight : string or None, optional (default=None)
       The edge attribute that holds the numerical value used as a weight.
       If None, then each edge has weight 1.
       The degree is the sum of the edge weights adjacent to the node.

    Returns
    -------
    (degX,degY) : tuple of dictionaries
       The degrees of the two bipartite sets as dictionaries keyed by node.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.complete_bipartite_graph(3, 2)
    >>> Y = set([3, 4])
    >>> degX, degY = bipartite.degrees(G, Y)
    >>> dict(degX)
    {0: 2, 1: 2, 2: 2}

    Notes
    -----
    The container of nodes passed as argument must contain all nodes
    in one of the two bipartite node sets to avoid ambiguity in the
    case of disconnected graphs.
    See :mod:`bipartite documentation <networkx.algorithms.bipartite>`
    for further details on how bipartite graphs are handled in NetworkX.

    See Also
    --------
    color, density
    """
    bottom = set(nodes)
    top = set(B) - bottom
    return (B.degree(top, weight), B.degree(bottom, weight))