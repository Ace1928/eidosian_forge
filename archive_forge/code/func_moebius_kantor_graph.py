from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def moebius_kantor_graph(create_using=None):
    """
    Returns the Moebius-Kantor graph.

    The Möbius-Kantor graph is the cubic symmetric graph on 16 nodes.
    Its LCF notation is [5,-5]^8, and it is isomorphic to the generalized
    Petersen graph [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Moebius-Kantor graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/M%C3%B6bius%E2%80%93Kantor_graph

    """
    G = LCF_graph(16, [5, -5], 8, create_using)
    G.name = 'Moebius-Kantor Graph'
    return G