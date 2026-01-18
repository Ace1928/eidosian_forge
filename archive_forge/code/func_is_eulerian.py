from itertools import combinations
import networkx as nx
from ..utils import arbitrary_element, not_implemented_for
@nx._dispatch
def is_eulerian(G):
    """Returns True if and only if `G` is Eulerian.

    A graph is *Eulerian* if it has an Eulerian circuit. An *Eulerian
    circuit* is a closed walk that includes each edge of a graph exactly
    once.

    Graphs with isolated vertices (i.e. vertices with zero degree) are not
    considered to have Eulerian circuits. Therefore, if the graph is not
    connected (or not strongly connected, for directed graphs), this function
    returns False.

    Parameters
    ----------
    G : NetworkX graph
       A graph, either directed or undirected.

    Examples
    --------
    >>> nx.is_eulerian(nx.DiGraph({0: [3], 1: [2], 2: [3], 3: [0, 1]}))
    True
    >>> nx.is_eulerian(nx.complete_graph(5))
    True
    >>> nx.is_eulerian(nx.petersen_graph())
    False

    If you prefer to allow graphs with isolated vertices to have Eulerian circuits,
    you can first remove such vertices and then call `is_eulerian` as below example shows.

    >>> G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    >>> G.add_node(3)
    >>> nx.is_eulerian(G)
    False

    >>> G.remove_nodes_from(list(nx.isolates(G)))
    >>> nx.is_eulerian(G)
    True


    """
    if G.is_directed():
        return all((G.in_degree(n) == G.out_degree(n) for n in G)) and nx.is_strongly_connected(G)
    return all((d % 2 == 0 for v, d in G.degree())) and nx.is_connected(G)