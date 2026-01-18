import warnings
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def average_shortest_path_length(G, weight=None, method=None):
    """Returns the average shortest path length.

    The average shortest path length is

    .. math::

       a =\\sum_{\\substack{s,t \\in V \\\\ s\\neq t}} \\frac{d(s, t)}{n(n-1)}

    where `V` is the set of nodes in `G`,
    `d(s, t)` is the shortest path from `s` to `t`,
    and `n` is the number of nodes in `G`.

    .. versionchanged:: 3.0
       An exception is raised for directed graphs that are not strongly
       connected.

    Parameters
    ----------
    G : NetworkX graph

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'unweighted' or 'dijkstra')
        The algorithm to use to compute the path lengths.
        Supported options are 'unweighted', 'dijkstra', 'bellman-ford',
        'floyd-warshall' and 'floyd-warshall-numpy'.
        Other method values produce a ValueError.
        The default method is 'unweighted' if `weight` is None,
        otherwise the default method is 'dijkstra'.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is the null graph (that is, the graph on zero nodes).

    NetworkXError
        If `G` is not connected (or not strongly connected, in the case
        of a directed graph).

    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.average_shortest_path_length(G)
    2.0

    For disconnected graphs, you can compute the average shortest path
    length for each component

    >>> G = nx.Graph([(1, 2), (3, 4)])
    >>> for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
    ...     print(nx.average_shortest_path_length(C))
    1.0
    1.0

    """
    single_source_methods = ['unweighted', 'dijkstra', 'bellman-ford']
    all_pairs_methods = ['floyd-warshall', 'floyd-warshall-numpy']
    supported_methods = single_source_methods + all_pairs_methods
    if method is None:
        method = 'unweighted' if weight is None else 'dijkstra'
    if method not in supported_methods:
        raise ValueError(f'method not supported: {method}')
    n = len(G)
    if n == 0:
        msg = 'the null graph has no paths, thus there is no average shortest path length'
        raise nx.NetworkXPointlessConcept(msg)
    if n == 1:
        return 0
    if G.is_directed() and (not nx.is_strongly_connected(G)):
        raise nx.NetworkXError('Graph is not strongly connected.')
    if not G.is_directed() and (not nx.is_connected(G)):
        raise nx.NetworkXError('Graph is not connected.')

    def path_length(v):
        if method == 'unweighted':
            return nx.single_source_shortest_path_length(G, v)
        elif method == 'dijkstra':
            return nx.single_source_dijkstra_path_length(G, v, weight=weight)
        elif method == 'bellman-ford':
            return nx.single_source_bellman_ford_path_length(G, v, weight=weight)
    if method in single_source_methods:
        s = sum((l for u in G for l in path_length(u).values()))
    elif method == 'floyd-warshall':
        all_pairs = nx.floyd_warshall(G, weight=weight)
        s = sum((sum(t.values()) for t in all_pairs.values()))
    elif method == 'floyd-warshall-numpy':
        s = nx.floyd_warshall_numpy(G, weight=weight).sum()
    return s / (n * (n - 1))