from collections import defaultdict
from itertools import combinations, permutations
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@nx._dispatch
def is_triad(G):
    """Returns True if the graph G is a triad, else False.

    Parameters
    ----------
    G : graph
       A NetworkX Graph

    Returns
    -------
    istriad : boolean
       Whether G is a valid triad

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    >>> nx.is_triad(G)
    True
    >>> G.add_edge(0, 1)
    >>> nx.is_triad(G)
    False
    """
    if isinstance(G, nx.Graph):
        if G.order() == 3 and nx.is_directed(G):
            if not any(((n, n) in G.edges() for n in G.nodes())):
                return True
    return False