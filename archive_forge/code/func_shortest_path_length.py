import warnings
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def shortest_path_length(G, source=None, target=None, weight=None, method='dijkstra'):
    """Compute shortest path lengths in the graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node, optional
        Starting node for path.
        If not specified, compute shortest path lengths using all nodes as
        source nodes.

    target : node, optional
        Ending node for path.
        If not specified, compute shortest path lengths using all nodes as
        target nodes.

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
        The algorithm to use to compute the path length.
        Supported options: 'dijkstra', 'bellman-ford'.
        Other inputs produce a ValueError.
        If `weight` is None, unweighted graph methods are used, and this
        suggestion is ignored.

    Returns
    -------
    length: int or iterator
        If the source and target are both specified, return the length of
        the shortest path from the source to the target.

        If only the source is specified, return a dict keyed by target
        to the shortest path length from the source to that target.

        If only the target is specified, return a dict keyed by source
        to the shortest path length from that source to the target.

        If neither the source nor target are specified, return an iterator
        over (source, dictionary) where dictionary is keyed by target to
        shortest path length from source to that target.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.shortest_path_length(G, source=0, target=4)
    4
    >>> p = nx.shortest_path_length(G, source=0)  # target not specified
    >>> p[4]
    4
    >>> p = nx.shortest_path_length(G, target=4)  # source not specified
    >>> p[0]
    4
    >>> p = dict(nx.shortest_path_length(G))  # source,target not specified
    >>> p[0][4]
    4

    Notes
    -----
    The length of the path is always 1 less than the number of nodes involved
    in the path since the length measures the number of edges followed.

    For digraphs this returns the shortest directed path length. To find path
    lengths in the reverse direction use G.reverse(copy=False) first to flip
    the edge orientation.

    See Also
    --------
    all_pairs_shortest_path_length
    all_pairs_dijkstra_path_length
    all_pairs_bellman_ford_path_length
    single_source_shortest_path_length
    single_source_dijkstra_path_length
    single_source_bellman_ford_path_length
    """
    if method not in ('dijkstra', 'bellman-ford'):
        raise ValueError(f'method not supported: {method}')
    method = 'unweighted' if weight is None else method
    if source is None:
        if target is None:
            if method == 'unweighted':
                paths = nx.all_pairs_shortest_path_length(G)
            elif method == 'dijkstra':
                paths = nx.all_pairs_dijkstra_path_length(G, weight=weight)
            else:
                paths = nx.all_pairs_bellman_ford_path_length(G, weight=weight)
        else:
            if G.is_directed():
                G = G.reverse(copy=False)
            if method == 'unweighted':
                path_length = nx.single_source_shortest_path_length
                paths = path_length(G, target)
            elif method == 'dijkstra':
                path_length = nx.single_source_dijkstra_path_length
                paths = path_length(G, target, weight=weight)
            else:
                path_length = nx.single_source_bellman_ford_path_length
                paths = path_length(G, target, weight=weight)
    elif target is None:
        if method == 'unweighted':
            paths = nx.single_source_shortest_path_length(G, source)
        elif method == 'dijkstra':
            path_length = nx.single_source_dijkstra_path_length
            paths = path_length(G, source, weight=weight)
        else:
            path_length = nx.single_source_bellman_ford_path_length
            paths = path_length(G, source, weight=weight)
    elif method == 'unweighted':
        p = nx.bidirectional_shortest_path(G, source, target)
        paths = len(p) - 1
    elif method == 'dijkstra':
        paths = nx.dijkstra_path_length(G, source, target, weight)
    else:
        paths = nx.bellman_ford_path_length(G, source, target, weight)
    return paths