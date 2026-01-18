import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nx._dispatch(graphs=None)
def ladder_graph(n, create_using=None):
    """Returns the Ladder graph of length n.

    This is two paths of n nodes, with
    each pair connected by a single edge.

    Node labels are the integers 0 to 2*n - 1.

    """
    G = empty_graph(2 * n, create_using)
    if G.is_directed():
        raise NetworkXError('Directed Graph not supported')
    G.add_edges_from(pairwise(range(n)))
    G.add_edges_from(pairwise(range(n, 2 * n)))
    G.add_edges_from(((v, v + n) for v in range(n)))
    return G