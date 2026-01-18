from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
@nx._dispatch(edge_attrs='weight')
def find_negative_cycle(G, source, weight='weight'):
    """Returns a cycle with negative total weight if it exists.

    Bellman-Ford is used to find shortest_paths. That algorithm
    stops if there exists a negative cycle. This algorithm
    picks up from there and returns the found negative cycle.

    The cycle consists of a list of nodes in the cycle order. The last
    node equals the first to make it a cycle.
    You can look up the edge weights in the original graph. In the case
    of multigraphs the relevant edge is the minimal weight edge between
    the nodes in the 2-tuple.

    If the graph has no negative cycle, a NetworkXError is raised.

    Parameters
    ----------
    G : NetworkX graph

    source: node label
        The search for the negative cycle will start from this node.

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

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_weighted_edges_from([(0, 1, 2), (1, 2, 2), (2, 0, 1), (1, 4, 2), (4, 0, -5)])
    >>> nx.find_negative_cycle(G, 0)
    [4, 0, 1, 4]

    Returns
    -------
    cycle : list
        A list of nodes in the order of the cycle found. The last node
        equals the first to indicate a cycle.

    Raises
    ------
    NetworkXError
        If no negative cycle is found.
    """
    weight = _weight_function(G, weight)
    pred = {source: []}
    v = _inner_bellman_ford(G, [source], weight, pred=pred)
    if v is None:
        raise nx.NetworkXError('No negative cycles detected.')
    neg_cycle = []
    stack = [(v, list(pred[v]))]
    seen = {v}
    while stack:
        node, preds = stack[-1]
        if v in preds:
            neg_cycle.extend([node, v])
            neg_cycle = list(reversed(neg_cycle))
            return neg_cycle
        if preds:
            nbr = preds.pop()
            if nbr not in seen:
                stack.append((nbr, list(pred[nbr])))
                neg_cycle.append(node)
                seen.add(nbr)
        else:
            stack.pop()
            if neg_cycle:
                neg_cycle.pop()
            else:
                if v in G[v] and weight(G, v, v) < 0:
                    return [v, v]
                raise nx.NetworkXError('Negative cycle is detected but not found')
    msg = 'negative cycle detected but not identified'
    raise nx.NetworkXUnbounded(msg)