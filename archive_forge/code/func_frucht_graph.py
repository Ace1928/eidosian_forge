from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def frucht_graph(create_using=None):
    """
    Returns the Frucht Graph.

    The Frucht Graph is the smallest cubical graph whose
    automorphism group consists only of the identity element [1]_.
    It has 12 nodes and 18 edges and no nontrivial symmetries.
    It is planar and Hamiltonian [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Frucht Graph with 12 nodes and 18 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Frucht_graph
    .. [2] https://mathworld.wolfram.com/FruchtGraph.html

    """
    G = cycle_graph(7, create_using)
    G.add_edges_from([[0, 7], [1, 7], [2, 8], [3, 9], [4, 9], [5, 10], [6, 10], [7, 11], [8, 11], [8, 9], [10, 11]])
    G.name = 'Frucht Graph'
    return G