from itertools import repeat
from math import sqrt
import networkx as nx
from networkx.classes import set_node_attributes
from networkx.exception import NetworkXError
from networkx.generators.classic import cycle_graph, empty_graph, path_graph
from networkx.relabel import relabel_nodes
from networkx.utils import flatten, nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def hypercube_graph(n):
    """Returns the *n*-dimensional hypercube graph.

    The nodes are the integers between 0 and ``2 ** n - 1``, inclusive.

    For more information on the hypercube graph, see the Wikipedia
    article `Hypercube graph`_.

    .. _Hypercube graph: https://en.wikipedia.org/wiki/Hypercube_graph

    Parameters
    ----------
    n : int
        The dimension of the hypercube.
        The number of nodes in the graph will be ``2 ** n``.

    Returns
    -------
    NetworkX graph
        The hypercube graph of dimension *n*.
    """
    dim = n * [2]
    G = grid_graph(dim)
    return G