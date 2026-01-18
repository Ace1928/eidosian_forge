from operator import itemgetter
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def newman_betweenness_centrality(G, v=None, cutoff=None, normalized=True, weight=None):
    """Compute load centrality for nodes.

    The load centrality of a node is the fraction of all shortest
    paths that pass through that node.

    Parameters
    ----------
    G : graph
      A networkx graph.

    normalized : bool, optional (default=True)
      If True the betweenness values are normalized by b=b/(n-1)(n-2) where
      n is the number of nodes in G.

    weight : None or string, optional (default=None)
      If None, edge weights are ignored.
      Otherwise holds the name of the edge attribute used as weight.
      The weight of an edge is treated as the length or distance between the two sides.

    cutoff : bool, optional (default=None)
      If specified, only consider paths of length <= cutoff.

    Returns
    -------
    nodes : dictionary
       Dictionary of nodes with centrality as the value.

    See Also
    --------
    betweenness_centrality

    Notes
    -----
    Load centrality is slightly different than betweenness. It was originally
    introduced by [2]_. For this load algorithm see [1]_.

    References
    ----------
    .. [1] Mark E. J. Newman:
       Scientific collaboration networks. II.
       Shortest paths, weighted networks, and centrality.
       Physical Review E 64, 016132, 2001.
       http://journals.aps.org/pre/abstract/10.1103/PhysRevE.64.016132
    .. [2] Kwang-Il Goh, Byungnam Kahng and Doochul Kim
       Universal behavior of Load Distribution in Scale-Free Networks.
       Physical Review Letters 87(27):1–4, 2001.
       https://doi.org/10.1103/PhysRevLett.87.278701
    """
    if v is not None:
        betweenness = 0.0
        for source in G:
            ubetween = _node_betweenness(G, source, cutoff, False, weight)
            betweenness += ubetween[v] if v in ubetween else 0
        if normalized:
            order = G.order()
            if order <= 2:
                return betweenness
            betweenness *= 1.0 / ((order - 1) * (order - 2))
    else:
        betweenness = {}.fromkeys(G, 0.0)
        for source in betweenness:
            ubetween = _node_betweenness(G, source, cutoff, False, weight)
            for vk in ubetween:
                betweenness[vk] += ubetween[vk]
        if normalized:
            order = G.order()
            if order <= 2:
                return betweenness
            scale = 1.0 / ((order - 1) * (order - 2))
            for v in betweenness:
                betweenness[v] *= scale
    return betweenness