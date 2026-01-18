import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.exception import AmbiguousSolution
@nx._dispatch
def is_bipartite_node_set(G, nodes):
    """Returns True if nodes and G/nodes are a bipartition of G.

    Parameters
    ----------
    G : NetworkX graph

    nodes: list or container
      Check if nodes are a one of a bipartite set.

    Examples
    --------
    >>> from networkx.algorithms import bipartite
    >>> G = nx.path_graph(4)
    >>> X = set([1, 3])
    >>> bipartite.is_bipartite_node_set(G, X)
    True

    Notes
    -----
    An exception is raised if the input nodes are not distinct, because in this
    case some bipartite algorithms will yield incorrect results.
    For connected graphs the bipartite sets are unique.  This function handles
    disconnected graphs.
    """
    S = set(nodes)
    if len(S) < len(nodes):
        raise AmbiguousSolution('The input node set contains duplicates.\nThis may lead to incorrect results when using it in bipartite algorithms.\nConsider using set(nodes) as the input')
    for CC in (G.subgraph(c).copy() for c in connected_components(G)):
        X, Y = sets(CC)
        if not (X.issubset(S) and Y.isdisjoint(S) or (Y.issubset(S) and X.isdisjoint(S))):
            return False
    return True