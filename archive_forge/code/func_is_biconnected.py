from itertools import chain
import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('directed')
@nx._dispatch
def is_biconnected(G):
    """Returns True if the graph is biconnected, False otherwise.

    A graph is biconnected if, and only if, it cannot be disconnected by
    removing only one node (and all edges incident on that node).  If
    removing a node increases the number of disconnected components
    in the graph, that node is called an articulation point, or cut
    vertex.  A biconnected graph has no articulation points.

    Parameters
    ----------
    G : NetworkX Graph
        An undirected graph.

    Returns
    -------
    biconnected : bool
        True if the graph is biconnected, False otherwise.

    Raises
    ------
    NetworkXNotImplemented
        If the input graph is not undirected.

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> print(nx.is_biconnected(G))
    False
    >>> G.add_edge(0, 3)
    >>> print(nx.is_biconnected(G))
    True

    See Also
    --------
    biconnected_components
    articulation_points
    biconnected_component_edges
    is_strongly_connected
    is_weakly_connected
    is_connected
    is_semiconnected

    Notes
    -----
    The algorithm to find articulation points and biconnected
    components is implemented using a non-recursive depth-first-search
    (DFS) that keeps track of the highest level that back edges reach
    in the DFS tree.  A node `n` is an articulation point if, and only
    if, there exists a subtree rooted at `n` such that there is no
    back edge from any successor of `n` that links to a predecessor of
    `n` in the DFS tree.  By keeping track of all the edges traversed
    by the DFS we can obtain the biconnected components because all
    edges of a bicomponent will be traversed consecutively between
    articulation points.

    References
    ----------
    .. [1] Hopcroft, J.; Tarjan, R. (1973).
       "Efficient algorithms for graph manipulation".
       Communications of the ACM 16: 372–378. doi:10.1145/362248.362272

    """
    bccs = biconnected_components(G)
    try:
        bcc = next(bccs)
    except StopIteration:
        return False
    try:
        next(bccs)
    except StopIteration:
        return len(bcc) == len(G)
    else:
        return False