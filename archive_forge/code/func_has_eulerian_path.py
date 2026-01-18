from itertools import combinations
import networkx as nx
from ..utils import arbitrary_element, not_implemented_for
@nx._dispatch
def has_eulerian_path(G, source=None):
    """Return True iff `G` has an Eulerian path.

    An Eulerian path is a path in a graph which uses each edge of a graph
    exactly once. If `source` is specified, then this function checks
    whether an Eulerian path that starts at node `source` exists.

    A directed graph has an Eulerian path iff:
        - at most one vertex has out_degree - in_degree = 1,
        - at most one vertex has in_degree - out_degree = 1,
        - every other vertex has equal in_degree and out_degree,
        - and all of its vertices belong to a single connected
          component of the underlying undirected graph.

    If `source` is not None, an Eulerian path starting at `source` exists if no
    other node has out_degree - in_degree = 1. This is equivalent to either
    there exists an Eulerian circuit or `source` has out_degree - in_degree = 1
    and the conditions above hold.

    An undirected graph has an Eulerian path iff:
        - exactly zero or two vertices have odd degree,
        - and all of its vertices belong to a single connected component.

    If `source` is not None, an Eulerian path starting at `source` exists if
    either there exists an Eulerian circuit or `source` has an odd degree and the
    conditions above hold.

    Graphs with isolated vertices (i.e. vertices with zero degree) are not considered
    to have an Eulerian path. Therefore, if the graph is not connected (or not strongly
    connected, for directed graphs), this function returns False.

    Parameters
    ----------
    G : NetworkX Graph
        The graph to find an euler path in.

    source : node, optional
        Starting node for path.

    Returns
    -------
    Bool : True if G has an Eulerian path.

    Examples
    --------
    If you prefer to allow graphs with isolated vertices to have Eulerian path,
    you can first remove such vertices and then call `has_eulerian_path` as below example shows.

    >>> G = nx.Graph([(0, 1), (1, 2), (0, 2)])
    >>> G.add_node(3)
    >>> nx.has_eulerian_path(G)
    False

    >>> G.remove_nodes_from(list(nx.isolates(G)))
    >>> nx.has_eulerian_path(G)
    True

    See Also
    --------
    is_eulerian
    eulerian_path
    """
    if nx.is_eulerian(G):
        return True
    if G.is_directed():
        ins = G.in_degree
        outs = G.out_degree
        if source is not None and outs[source] - ins[source] != 1:
            return False
        unbalanced_ins = 0
        unbalanced_outs = 0
        for v in G:
            if ins[v] - outs[v] == 1:
                unbalanced_ins += 1
            elif outs[v] - ins[v] == 1:
                unbalanced_outs += 1
            elif ins[v] != outs[v]:
                return False
        return unbalanced_ins <= 1 and unbalanced_outs <= 1 and nx.is_weakly_connected(G)
    else:
        if source is not None and G.degree[source] % 2 != 1:
            return False
        return sum((d % 2 == 1 for v, d in G.degree())) == 2 and nx.is_connected(G)