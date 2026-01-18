import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
@nodes_or_number(0)
@nx._dispatch(graphs=None)
def complete_graph(n, create_using=None):
    """Return the complete graph `K_n` with n nodes.

    A complete graph on `n` nodes means that all pairs
    of distinct nodes have an edge connecting them.

    Parameters
    ----------
    n : int or iterable container of nodes
        If n is an integer, nodes are from range(n).
        If n is a container of nodes, those nodes appear in the graph.
        Warning: n is not checked for duplicates and if present the
        resulting graph may not be as desired. Make sure you have no duplicates.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Examples
    --------
    >>> G = nx.complete_graph(9)
    >>> len(G)
    9
    >>> G.size()
    36
    >>> G = nx.complete_graph(range(11, 14))
    >>> list(G.nodes())
    [11, 12, 13]
    >>> G = nx.complete_graph(4, nx.DiGraph())
    >>> G.is_directed()
    True

    """
    _, nodes = n
    G = empty_graph(nodes, create_using)
    if len(nodes) > 1:
        if G.is_directed():
            edges = itertools.permutations(nodes, 2)
        else:
            edges = itertools.combinations(nodes, 2)
        G.add_edges_from(edges)
    return G