import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nodes_or_number(0)
@nx._dispatch(graphs=None)
def cycle_graph(n, create_using=None):
    """Returns the cycle graph $C_n$ of cyclically connected nodes.

    $C_n$ is a path with its two end-nodes connected.

    Parameters
    ----------
    n : int or iterable container of nodes
        If n is an integer, nodes are from `range(n)`.
        If n is a container of nodes, those nodes appear in the graph.
        Warning: n is not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Notes
    -----
    If create_using is directed, the direction is in increasing order.

    """
    _, nodes = n
    G = empty_graph(nodes, create_using)
    G.add_edges_from(pairwise(nodes, cyclic=True))
    return G