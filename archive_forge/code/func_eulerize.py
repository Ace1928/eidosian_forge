from itertools import combinations
import networkx as nx
from ..utils import arbitrary_element, not_implemented_for
@not_implemented_for('directed')
@nx._dispatch
def eulerize(G):
    """Transforms a graph into an Eulerian graph.

    If `G` is Eulerian the result is `G` as a MultiGraph, otherwise the result is a smallest
    (in terms of the number of edges) multigraph whose underlying simple graph is `G`.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph

    Returns
    -------
    G : NetworkX multigraph

    Raises
    ------
    NetworkXError
       If the graph is not connected.

    See Also
    --------
    is_eulerian
    eulerian_circuit

    References
    ----------
    .. [1] J. Edmonds, E. L. Johnson.
       Matching, Euler tours and the Chinese postman.
       Mathematical programming, Volume 5, Issue 1 (1973), 111-114.
    .. [2] https://en.wikipedia.org/wiki/Eulerian_path
    .. [3] http://web.math.princeton.edu/math_alive/5/Notes1.pdf

    Examples
    --------
        >>> G = nx.complete_graph(10)
        >>> H = nx.eulerize(G)
        >>> nx.is_eulerian(H)
        True

    """
    if G.order() == 0:
        raise nx.NetworkXPointlessConcept('Cannot Eulerize null graph')
    if not nx.is_connected(G):
        raise nx.NetworkXError('G is not connected')
    odd_degree_nodes = [n for n, d in G.degree() if d % 2 == 1]
    G = nx.MultiGraph(G)
    if len(odd_degree_nodes) == 0:
        return G
    odd_deg_pairs_paths = [(m, {n: nx.shortest_path(G, source=m, target=n)}) for m, n in combinations(odd_degree_nodes, 2)]
    upper_bound_on_max_path_length = len(G) + 1
    Gp = nx.Graph()
    for n, Ps in odd_deg_pairs_paths:
        for m, P in Ps.items():
            if n != m:
                Gp.add_edge(m, n, weight=upper_bound_on_max_path_length - len(P), path=P)
    best_matching = nx.Graph(list(nx.max_weight_matching(Gp)))
    for m, n in best_matching.edges():
        path = Gp[m][n]['path']
        G.add_edges_from(nx.utils.pairwise(path))
    return G