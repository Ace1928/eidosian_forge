from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def house_x_graph(create_using=None):
    """
    Returns the House graph with a cross inside the house square.

    The House X-graph is the House graph plus the two edges connecting diagonally
    opposite vertices of the square base. It is also one of the two graphs
    obtained by removing two edges from the pentatope graph [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        House graph with diagonal vertices connected

    References
    ----------
    .. [1] https://mathworld.wolfram.com/HouseGraph.html
    """
    G = house_graph(create_using)
    G.add_edges_from([(0, 3), (1, 2)])
    G.name = 'House-with-X-inside Graph'
    return G