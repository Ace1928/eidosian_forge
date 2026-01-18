from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def house_graph(create_using=None):
    """
    Returns the House graph (square with triangle on top)

    The house graph is a simple undirected graph with
    5 nodes and 6 edges [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        House graph in the form of a square with a triangle on top

    References
    ----------
    .. [1] https://mathworld.wolfram.com/HouseGraph.html
    """
    G = nx.from_dict_of_lists({0: [1, 2], 1: [0, 3], 2: [0, 3, 4], 3: [1, 2, 4], 4: [2, 3]}, create_using=create_using)
    G.name = 'House Graph'
    return G