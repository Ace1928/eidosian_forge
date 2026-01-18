from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def desargues_graph(create_using=None):
    """
    Returns the Desargues Graph

    The Desargues Graph is a non-planar, distance-transitive cubic graph
    with 20 nodes and 30 edges [1]_.
    It is a symmetric graph. It can be represented in LCF notation
    as [5,-5,9,-9]^5 [2]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Desargues Graph with 20 nodes and 30 edges

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Desargues_graph
    .. [2] https://mathworld.wolfram.com/DesarguesGraph.html
    """
    G = LCF_graph(20, [5, -5, 9, -9], 5, create_using)
    G.name = 'Desargues Graph'
    return G