import networkx as nx
from networkx.utils import pairwise
@nx._dispatch(edge_attrs='weight')
def local_reaching_centrality(G, v, paths=None, weight=None, normalized=True):
    """Returns the local reaching centrality of a node in a directed
    graph.

    The *local reaching centrality* of a node in a directed graph is the
    proportion of other nodes reachable from that node [1]_.

    Parameters
    ----------
    G : DiGraph
        A NetworkX DiGraph.

    v : node
        A node in the directed graph `G`.

    paths : dictionary (default=None)
        If this is not `None` it must be a dictionary representation
        of single-source shortest paths, as computed by, for example,
        :func:`networkx.shortest_path` with source node `v`. Use this
        keyword argument if you intend to invoke this function many
        times but don't want the paths to be recomputed each time.

    weight : None or string, optional (default=None)
        Attribute to use for edge weights.  If `None`, each edge weight
        is assumed to be one. A higher weight implies a stronger
        connection between nodes and a *shorter* path length.

    normalized : bool, optional (default=True)
        Whether to normalize the edge weights by the total sum of edge
        weights.

    Returns
    -------
    h : float
        The local reaching centrality of the node ``v`` in the graph
        ``G``.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3)])
    >>> nx.local_reaching_centrality(G, 3)
    0.0
    >>> G.add_edge(3, 2)
    >>> nx.local_reaching_centrality(G, 3)
    0.5

    See also
    --------
    global_reaching_centrality

    References
    ----------
    .. [1] Mones, Enys, Lilla Vicsek, and Tamás Vicsek.
           "Hierarchy Measure for Complex Networks."
           *PLoS ONE* 7.3 (2012): e33799.
           https://doi.org/10.1371/journal.pone.0033799
    """
    if paths is None:
        if nx.is_negatively_weighted(G, weight=weight):
            raise nx.NetworkXError('edge weights must be positive')
        total_weight = G.size(weight=weight)
        if total_weight <= 0:
            raise nx.NetworkXError('Size of G must be positive')
        if weight is not None:

            def as_distance(u, v, d):
                return total_weight / d.get(weight, 1)
            paths = nx.shortest_path(G, source=v, weight=as_distance)
        else:
            paths = nx.shortest_path(G, source=v)
    if weight is None and G.is_directed():
        return (len(paths) - 1) / (len(G) - 1)
    if normalized and weight is not None:
        norm = G.size(weight=weight) / G.size()
    else:
        norm = 1
    avgw = (_average_weight(G, path, weight=weight) for path in paths.values())
    sum_avg_weight = sum(avgw) / norm
    return sum_avg_weight / (len(G) - 1)