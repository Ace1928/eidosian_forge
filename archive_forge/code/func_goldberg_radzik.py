from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
@nx._dispatch(edge_attrs='weight')
def goldberg_radzik(G, source, weight='weight'):
    """Compute shortest path lengths and predecessors on shortest paths
    in weighted graphs.

    The algorithm has a running time of $O(mn)$ where $n$ is the number of
    nodes and $m$ is the number of edges.  It is slower than Dijkstra but
    can handle negative edge weights.

    Parameters
    ----------
    G : NetworkX graph
        The algorithm works for all types of graphs, including directed
        graphs and multigraphs.

    source: node label
        Starting node for path

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
    pred, dist : dictionaries
        Returns two dictionaries keyed by node to predecessor in the
        path and to the distance from the source respectively.

    Raises
    ------
    NodeNotFound
        If `source` is not in `G`.

    NetworkXUnbounded
        If the (di)graph contains a negative (di)cycle, the
        algorithm raises an exception to indicate the presence of the
        negative (di)cycle.  Note: any negative weight edge in an
        undirected graph is a negative cycle.

        As of NetworkX v3.2, a zero weight cycle is no longer
        incorrectly reported as a negative weight cycle.


    Examples
    --------
    >>> G = nx.path_graph(5, create_using=nx.DiGraph())
    >>> pred, dist = nx.goldberg_radzik(G, 0)
    >>> sorted(pred.items())
    [(0, None), (1, 0), (2, 1), (3, 2), (4, 3)]
    >>> sorted(dist.items())
    [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]

    >>> G = nx.cycle_graph(5, create_using=nx.DiGraph())
    >>> G[1][2]["weight"] = -7
    >>> nx.goldberg_radzik(G, 0)
    Traceback (most recent call last):
        ...
    networkx.exception.NetworkXUnbounded: Negative cycle detected.

    Notes
    -----
    Edge weight attributes must be numerical.
    Distances are calculated as sums of weighted edges traversed.

    The dictionaries returned only have keys for nodes reachable from
    the source.

    In the case where the (di)graph is not connected, if a component
    not containing the source contains a negative (di)cycle, it
    will not be detected.

    """
    if source not in G:
        raise nx.NodeNotFound(f'Node {source} is not found in the graph')
    weight = _weight_function(G, weight)
    if G.is_multigraph():
        if any((weight(u, v, {k: d}) < 0 for u, v, k, d in nx.selfloop_edges(G, keys=True, data=True))):
            raise nx.NetworkXUnbounded('Negative cycle detected.')
    elif any((weight(u, v, d) < 0 for u, v, d in nx.selfloop_edges(G, data=True))):
        raise nx.NetworkXUnbounded('Negative cycle detected.')
    if len(G) == 1:
        return ({source: None}, {source: 0})
    G_succ = G._adj
    inf = float('inf')
    d = {u: inf for u in G}
    d[source] = 0
    pred = {source: None}

    def topo_sort(relabeled):
        """Topologically sort nodes relabeled in the previous round and detect
        negative cycles.
        """
        to_scan = []
        neg_count = {}
        for u in relabeled:
            if u in neg_count:
                continue
            d_u = d[u]
            if all((d_u + weight(u, v, e) >= d[v] for v, e in G_succ[u].items())):
                continue
            stack = [(u, iter(G_succ[u].items()))]
            in_stack = {u}
            neg_count[u] = 0
            while stack:
                u, it = stack[-1]
                try:
                    v, e = next(it)
                except StopIteration:
                    to_scan.append(u)
                    stack.pop()
                    in_stack.remove(u)
                    continue
                t = d[u] + weight(u, v, e)
                d_v = d[v]
                if t < d_v:
                    is_neg = t < d_v
                    d[v] = t
                    pred[v] = u
                    if v not in neg_count:
                        neg_count[v] = neg_count[u] + int(is_neg)
                        stack.append((v, iter(G_succ[v].items())))
                        in_stack.add(v)
                    elif v in in_stack and neg_count[u] + int(is_neg) > neg_count[v]:
                        raise nx.NetworkXUnbounded('Negative cycle detected.')
        to_scan.reverse()
        return to_scan

    def relax(to_scan):
        """Relax out-edges of relabeled nodes."""
        relabeled = set()
        for u in to_scan:
            d_u = d[u]
            for v, e in G_succ[u].items():
                w_e = weight(u, v, e)
                if d_u + w_e < d[v]:
                    d[v] = d_u + w_e
                    pred[v] = u
                    relabeled.add(v)
        return relabeled
    relabeled = {source}
    while relabeled:
        to_scan = topo_sort(relabeled)
        relabeled = relax(to_scan)
    d = {u: d[u] for u in pred}
    return (pred, d)