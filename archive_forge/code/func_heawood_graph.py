from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def heawood_graph(create_using=None):
    """
    Returns the Heawood Graph, a (3,6) cage.

    The Heawood Graph is an undirected graph with 14 nodes and 21 edges,
    named after Percy John Heawood [1]_.
    It is cubic symmetric, nonplanar, Hamiltonian, and can be represented
    in LCF notation as ``[5,-5]^7`` [2]_.
    It is the unique (3,6)-cage: the regular cubic graph of girth 6 with
    minimal number of vertices [3]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Heawood Graph with 14 nodes and 21 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Heawood_graph
    .. [2] https://mathworld.wolfram.com/HeawoodGraph.html
    .. [3] https://www.win.tue.nl/~aeb/graphs/Heawood.html

    """
    G = LCF_graph(14, [5, -5], 7, create_using)
    G.name = 'Heawood Graph'
    return G