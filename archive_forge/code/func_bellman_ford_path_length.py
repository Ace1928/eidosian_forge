from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
@nx._dispatch(edge_attrs='weight')
def bellman_ford_path_length(G, source, target, weight='weight'):
    """Returns the shortest path length from source to target
    in a weighted graph.

    Parameters
    ----------
    G : NetworkX graph

    source : node label
        starting node for path

    target : node label
        ending node for path

    weight : string or function (default="weight")
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
    length : number
        Shortest path length.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXNoPath
        If no path exists between source and target.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.bellman_ford_path_length(G, 0, 4)
    4

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    See Also
    --------
    dijkstra_path_length, bellman_ford_path
    """
    if source == target:
        if source not in G:
            raise nx.NodeNotFound(f'Node {source} not found in graph')
        return 0
    weight = _weight_function(G, weight)
    length = _bellman_ford(G, [source], weight, target=target)
    try:
        return length[target]
    except KeyError as err:
        raise nx.NetworkXNoPath(f'node {target} not reachable from {source}') from err