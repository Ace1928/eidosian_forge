from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
@nx._dispatch(edge_attrs='weight')
def johnson(G, weight='weight'):
    """Uses Johnson's Algorithm to compute shortest paths.

    Johnson's Algorithm finds a shortest path between each pair of
    nodes in a weighted graph even if negative weights are present.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

    Returns
    -------
    distance : dictionary
        Dictionary, keyed by source and target, of shortest paths.

    Examples
    --------
    >>> graph = nx.DiGraph()
    >>> graph.add_weighted_edges_from(
    ...     [("0", "3", 3), ("0", "1", -5), ("0", "2", 2), ("1", "2", 4), ("2", "3", 1)]
    ... )
    >>> paths = nx.johnson(graph, weight="weight")
    >>> paths["0"]["2"]
    ['0', '1', '2']

    Notes
    -----
    Johnson's algorithm is suitable even for graphs with negative weights. It
    works by using the Bellman–Ford algorithm to compute a transformation of
    the input graph that removes all negative weights, allowing Dijkstra's
    algorithm to be used on the transformed graph.

    The time complexity of this algorithm is $O(n^2 \\log n + n m)$,
    where $n$ is the number of nodes and $m$ the number of edges in the
    graph. For dense graphs, this may be faster than the Floyd–Warshall
    algorithm.

    See Also
    --------
    floyd_warshall_predecessor_and_distance
    floyd_warshall_numpy
    all_pairs_shortest_path
    all_pairs_shortest_path_length
    all_pairs_dijkstra_path
    bellman_ford_predecessor_and_distance
    all_pairs_bellman_ford_path
    all_pairs_bellman_ford_path_length

    """
    dist = {v: 0 for v in G}
    pred = {v: [] for v in G}
    weight = _weight_function(G, weight)
    dist_bellman = _bellman_ford(G, list(G), weight, pred=pred, dist=dist)

    def new_weight(u, v, d):
        return weight(u, v, d) + dist_bellman[u] - dist_bellman[v]

    def dist_path(v):
        paths = {v: [v]}
        _dijkstra(G, v, new_weight, paths=paths)
        return paths
    return {v: dist_path(v) for v in G}