import networkx as nx
from networkx.exception import NetworkXNoPath
from ..utils import not_implemented_for
@not_implemented_for('directed')
@nx._dispatch
def global_efficiency(G):
    """Returns the average global efficiency of the graph.

    The *efficiency* of a pair of nodes in a graph is the multiplicative
    inverse of the shortest path distance between the nodes. The *average
    global efficiency* of a graph is the average efficiency of all pairs of
    nodes [1]_.

    Parameters
    ----------
    G : :class:`networkx.Graph`
        An undirected graph for which to compute the average global efficiency.

    Returns
    -------
    float
        The average global efficiency of the graph.

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)])
    >>> round(nx.global_efficiency(G), 12)
    0.916666666667

    Notes
    -----
    Edge weights are ignored when computing the shortest path distances.

    See also
    --------
    local_efficiency

    References
    ----------
    .. [1] Latora, Vito, and Massimo Marchiori.
           "Efficient behavior of small-world networks."
           *Physical Review Letters* 87.19 (2001): 198701.
           <https://doi.org/10.1103/PhysRevLett.87.198701>

    """
    n = len(G)
    denom = n * (n - 1)
    if denom != 0:
        lengths = nx.all_pairs_shortest_path_length(G)
        g_eff = 0
        for source, targets in lengths:
            for target, distance in targets.items():
                if distance > 0:
                    g_eff += 1 / distance
        g_eff /= denom
    else:
        g_eff = 0
    return g_eff