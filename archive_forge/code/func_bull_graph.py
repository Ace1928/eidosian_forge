from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def bull_graph(create_using=None):
    """
    Returns the Bull Graph

    The Bull Graph has 5 nodes and 5 edges. It is a planar undirected
    graph in the form of a triangle with two disjoint pendant edges [1]_
    The name comes from the triangle and pendant edges representing
    respectively the body and legs of a bull.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        A bull graph with 5 nodes

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bull_graph.

    """
    G = nx.from_dict_of_lists({0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 4], 3: [1], 4: [2]}, create_using=create_using)
    G.name = 'Bull Graph'
    return G