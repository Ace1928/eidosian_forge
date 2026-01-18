import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def circular_ladder_graph(n, create_using=None):
    """Returns the circular ladder graph $CL_n$ of length n.

    $CL_n$ consists of two concentric n-cycles in which
    each of the n pairs of concentric nodes are joined by an edge.

    Node labels are the integers 0 to n-1

    """
    G = ladder_graph(n, create_using)
    G.add_edge(0, n - 1)
    G.add_edge(n, 2 * n - 1)
    return G