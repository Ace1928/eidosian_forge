from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def pappus_graph():
    """
    Returns the Pappus graph.

    The Pappus graph is a cubic symmetric distance-regular graph with 18 nodes
    and 27 edges. It is Hamiltonian and can be represented in LCF notation as
    [5,7,-7,7,-7,-5]^3 [1]_.

    Returns
    -------
    G : networkx Graph
        Pappus graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pappus_graph
    """
    G = LCF_graph(18, [5, 7, -7, 7, -7, -5], 3)
    G.name = 'Pappus Graph'
    return G