import sys
import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.utils import arbitrary_element, not_implemented_for
@nx._dispatch
def find_induced_nodes(G, s, t, treewidth_bound=sys.maxsize):
    """Returns the set of induced nodes in the path from s to t.

    Parameters
    ----------
    G : graph
      A chordal NetworkX graph
    s : node
        Source node to look for induced nodes
    t : node
        Destination node to look for induced nodes
    treewidth_bound: float
        Maximum treewidth acceptable for the graph H. The search
        for induced nodes will end as soon as the treewidth_bound is exceeded.

    Returns
    -------
    induced_nodes : Set of nodes
        The set of induced nodes in the path from s to t in G

    Raises
    ------
    NetworkXError
        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.
        If the input graph is an instance of one of these classes, a
        :exc:`NetworkXError` is raised.
        The algorithm can only be applied to chordal graphs. If the input
        graph is found to be non-chordal, a :exc:`NetworkXError` is raised.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G = nx.generators.classic.path_graph(10)
    >>> induced_nodes = nx.find_induced_nodes(G, 1, 9, 2)
    >>> sorted(induced_nodes)
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    Notes
    -----
    G must be a chordal graph and (s,t) an edge that is not in G.

    If a treewidth_bound is provided, the search for induced nodes will end
    as soon as the treewidth_bound is exceeded.

    The algorithm is inspired by Algorithm 4 in [1]_.
    A formal definition of induced node can also be found on that reference.

    Self Loops are ignored

    References
    ----------
    .. [1] Learning Bounded Treewidth Bayesian Networks.
       Gal Elidan, Stephen Gould; JMLR, 9(Dec):2699--2731, 2008.
       http://jmlr.csail.mit.edu/papers/volume9/elidan08a/elidan08a.pdf
    """
    if not is_chordal(G):
        raise nx.NetworkXError('Input graph is not chordal.')
    H = nx.Graph(G)
    H.add_edge(s, t)
    induced_nodes = set()
    triplet = _find_chordality_breaker(H, s, treewidth_bound)
    while triplet:
        u, v, w = triplet
        induced_nodes.update(triplet)
        for n in triplet:
            if n != s:
                H.add_edge(s, n)
        triplet = _find_chordality_breaker(H, s, treewidth_bound)
    if induced_nodes:
        induced_nodes.add(t)
        for u in G[s]:
            if len(induced_nodes & set(G[u])) == 2:
                induced_nodes.add(u)
                break
    return induced_nodes