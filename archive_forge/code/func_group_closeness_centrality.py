from copy import deepcopy
import networkx as nx
from networkx.algorithms.centrality.betweenness import (
from networkx.utils.decorators import not_implemented_for
@nx._dispatch(edge_attrs='weight')
def group_closeness_centrality(G, S, weight=None):
    """Compute the group closeness centrality for a group of nodes.

    Group closeness centrality of a group of nodes $S$ is a measure
    of how close the group is to the other nodes in the graph.

    .. math::

       c_{close}(S) = \\frac{|V-S|}{\\sum_{v \\in V-S} d_{S, v}}

       d_{S, v} = min_{u \\in S} (d_{u, v})

    where $V$ is the set of nodes, $d_{S, v}$ is the distance of
    the group $S$ from $v$ defined as above. ($V-S$ is the set of nodes
    in $V$ that are not in $S$).

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    S : list or set
       S is a group of nodes which belong to G, for which group closeness
       centrality is to be calculated.

    weight : None or string, optional (default=None)
       If None, all edge weights are considered equal.
       Otherwise holds the name of the edge attribute used as weight.
       The weight of an edge is treated as the length or distance between the two sides.

    Raises
    ------
    NodeNotFound
       If node(s) in S are not present in G.

    Returns
    -------
    closeness : float
       Group closeness centrality of the group S.

    See Also
    --------
    closeness_centrality

    Notes
    -----
    The measure was introduced in [1]_.
    The formula implemented here is described in [2]_.

    Higher values of closeness indicate greater centrality.

    It is assumed that 1 / 0 is 0 (required in the case of directed graphs,
    or when a shortest path length is 0).

    The number of nodes in the group must be a maximum of n - 1 where `n`
    is the total number of nodes in the graph.

    For directed graphs, the incoming distance is utilized here. To use the
    outward distance, act on `G.reverse()`.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] J. Zhao et. al.:
       Measuring and Maximizing Group Closeness Centrality over
       Disk Resident Graphs.
       WWWConference Proceedings, 2014. 689-694.
       https://doi.org/10.1145/2567948.2579356
    """
    if G.is_directed():
        G = G.reverse()
    closeness = 0
    V = set(G)
    S = set(S)
    V_S = V - S
    shortest_path_lengths = nx.multi_source_dijkstra_path_length(G, S, weight=weight)
    for v in V_S:
        try:
            closeness += shortest_path_lengths[v]
        except KeyError:
            closeness += 0
    try:
        closeness = len(V_S) / closeness
    except ZeroDivisionError:
        closeness = 0
    return closeness