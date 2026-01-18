from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def icosahedral_graph(create_using=None):
    """
    Returns the Platonic Icosahedral graph.

    The icosahedral graph has 12 nodes and 30 edges. It is a Platonic graph
    whose nodes have the connectivity of the icosahedron. It is undirected,
    regular and Hamiltonian [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Icosahedral graph with 12 nodes and 30 edges.

    References
    ----------
    .. [1] https://mathworld.wolfram.com/IcosahedralGraph.html
    """
    G = nx.from_dict_of_lists({0: [1, 5, 7, 8, 11], 1: [2, 5, 6, 8], 2: [3, 6, 8, 9], 3: [4, 6, 9, 10], 4: [5, 6, 10, 11], 5: [6, 11], 7: [8, 9, 10, 11], 8: [9], 9: [10], 10: [11]}, create_using=create_using)
    G.name = 'Platonic Icosahedral Graph'
    return G