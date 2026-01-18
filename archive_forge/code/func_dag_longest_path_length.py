import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@not_implemented_for('undirected')
@nx._dispatch(edge_attrs={'weight': 'default_weight'})
def dag_longest_path_length(G, weight='weight', default_weight=1):
    """Returns the longest path length in a DAG

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed acyclic graph (DAG)

    weight : string, optional
        Edge data key to use for weight

    default_weight : int, optional
        The weight of edges that do not have a weight attribute

    Returns
    -------
    int
        Longest path length

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed

    Examples
    --------
    >>> DG = nx.DiGraph([(0, 1, {'cost':1}), (1, 2, {'cost':1}), (0, 2, {'cost':42})])
    >>> list(nx.all_simple_paths(DG, 0, 2))
    [[0, 1, 2], [0, 2]]
    >>> nx.dag_longest_path_length(DG)
    2
    >>> nx.dag_longest_path_length(DG, weight="cost")
    42

    See also
    --------
    dag_longest_path
    """
    path = nx.dag_longest_path(G, weight, default_weight)
    path_length = 0
    if G.is_multigraph():
        for u, v in pairwise(path):
            i = max(G[u][v], key=lambda x: G[u][v][x].get(weight, default_weight))
            path_length += G[u][v][i].get(weight, default_weight)
    else:
        for u, v in pairwise(path):
            path_length += G[u][v].get(weight, default_weight)
    return path_length