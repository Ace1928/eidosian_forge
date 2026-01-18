from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def is_matching(G, matching):
    """Return True if ``matching`` is a valid matching of ``G``

    A *matching* in a graph is a set of edges in which no two distinct
    edges share a common endpoint. Each node is incident to at most one
    edge in the matching. The edges are said to be independent.

    Parameters
    ----------
    G : NetworkX graph

    matching : dict or set
        A dictionary or set representing a matching. If a dictionary, it
        must have ``matching[u] == v`` and ``matching[v] == u`` for each
        edge ``(u, v)`` in the matching. If a set, it must have elements
        of the form ``(u, v)``, where ``(u, v)`` is an edge in the
        matching.

    Returns
    -------
    bool
        Whether the given set or dictionary represents a valid matching
        in the graph.

    Raises
    ------
    NetworkXError
        If the proposed matching has an edge to a node not in G.
        Or if the matching is not a collection of 2-tuple edges.

    Examples
    --------
    >>> G = nx.Graph([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)])
    >>> nx.is_maximal_matching(G, {1: 3, 2: 4})  # using dict to represent matching
    True

    >>> nx.is_matching(G, {(1, 3), (2, 4)})  # using set to represent matching
    True

    """
    if isinstance(matching, dict):
        matching = matching_dict_to_set(matching)
    nodes = set()
    for edge in matching:
        if len(edge) != 2:
            raise nx.NetworkXError(f'matching has non-2-tuple edge {edge}')
        u, v = edge
        if u not in G or v not in G:
            raise nx.NetworkXError(f'matching contains edge {edge} with node not in G')
        if u == v:
            return False
        if not G.has_edge(u, v):
            return False
        if u in nodes or v in nodes:
            return False
        nodes.update(edge)
    return True