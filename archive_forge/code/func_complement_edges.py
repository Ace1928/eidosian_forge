import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@nx._dispatch
def complement_edges(G):
    """Returns only the edges in the complement of G

    Parameters
    ----------
    G : NetworkX Graph

    Yields
    ------
    edge : tuple
        Edges in the complement of G

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4))
    >>> sorted(complement_edges(G))
    [(1, 3), (1, 4), (2, 4)]
    >>> G = nx.path_graph((1, 2, 3, 4), nx.DiGraph())
    >>> sorted(complement_edges(G))
    [(1, 3), (1, 4), (2, 1), (2, 4), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)]
    >>> G = nx.complete_graph(1000)
    >>> sorted(complement_edges(G))
    []
    """
    G_adj = G._adj
    if G.is_directed():
        for u, v in it.combinations(G.nodes(), 2):
            if v not in G_adj[u]:
                yield (u, v)
            if u not in G_adj[v]:
                yield (v, u)
    else:
        for u, v in it.combinations(G.nodes(), 2):
            if v not in G_adj[u]:
                yield (u, v)