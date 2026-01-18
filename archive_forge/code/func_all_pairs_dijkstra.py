from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
@nx._dispatch(edge_attrs='weight')
def all_pairs_dijkstra(G, cutoff=None, weight='weight'):
    """Find shortest weighted paths and lengths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer or float, optional
        Length (sum of edge weights) at which the search is stopped.
        If cutoff is provided, only return paths with summed weight <= cutoff.

    weight : string or function
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edge[u][v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number or None to indicate a hidden edge.

    Yields
    ------
    (node, (distance, path)) : (node obj, (dict, dict))
        Each source node has two associated dicts. The first holds distance
        keyed by target and the second holds paths keyed by target.
        (See single_source_dijkstra for the source/target node terminology.)
        If desired you can apply `dict()` to this function to create a dict
        keyed by source node to the two dicts.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> len_path = dict(nx.all_pairs_dijkstra(G))
    >>> len_path[3][0][1]
    2
    >>> for node in [0, 1, 2, 3, 4]:
    ...     print(f"3 - {node}: {len_path[3][0][node]}")
    3 - 0: 3
    3 - 1: 2
    3 - 2: 1
    3 - 3: 0
    3 - 4: 1
    >>> len_path[3][1][1]
    [3, 2, 1]
    >>> for n, (dist, path) in nx.all_pairs_dijkstra(G):
    ...     print(path[1])
    [0, 1]
    [1]
    [2, 1]
    [3, 2, 1]
    [4, 3, 2, 1]

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The yielded dicts only have keys for reachable nodes.
    """
    for n in G:
        dist, path = single_source_dijkstra(G, n, cutoff=cutoff, weight=weight)
        yield (n, (dist, path))