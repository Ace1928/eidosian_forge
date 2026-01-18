from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def number_of_selfloops(G):
    """Returns the number of selfloop edges.

    A selfloop edge has the same node at both ends.

    Returns
    -------
    nloops : int
        The number of selfloops.

    See Also
    --------
    nodes_with_selfloops, selfloop_edges

    Examples
    --------
    >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
    >>> G.add_edge(1, 1)
    >>> G.add_edge(1, 2)
    >>> nx.number_of_selfloops(G)
    1
    """
    return sum((1 for _ in nx.selfloop_edges(G)))