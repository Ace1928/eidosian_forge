import math
import numbers
from functools import reduce
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
@nodes_or_number([0, 1])
@nx._dispatch(graphs=None)
def complete_bipartite_graph(n1, n2, create_using=None):
    """Returns the complete bipartite graph `K_{n_1,n_2}`.

    The graph is composed of two partitions with nodes 0 to (n1 - 1)
    in the first and nodes n1 to (n1 + n2 - 1) in the second.
    Each node in the first is connected to each node in the second.

    Parameters
    ----------
    n1, n2 : integer or iterable container of nodes
        If integers, nodes are from `range(n1)` and `range(n1, n1 + n2)`.
        If a container, the elements are the nodes.
    create_using : NetworkX graph instance, (default: nx.Graph)
       Return graph of this type.

    Notes
    -----
    Nodes are the integers 0 to `n1 + n2 - 1` unless either n1 or n2 are
    containers of nodes. If only one of n1 or n2 are integers, that
    integer is replaced by `range` of that integer.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.complete_bipartite_graph
    """
    G = nx.empty_graph(0, create_using)
    if G.is_directed():
        raise nx.NetworkXError('Directed Graph not supported')
    n1, top = n1
    n2, bottom = n2
    if isinstance(n1, numbers.Integral) and isinstance(n2, numbers.Integral):
        bottom = [n1 + i for i in bottom]
    G.add_nodes_from(top, bipartite=0)
    G.add_nodes_from(bottom, bipartite=1)
    if len(G) != len(top) + len(bottom):
        raise nx.NetworkXError('Inputs n1 and n2 must contain distinct nodes')
    G.add_edges_from(((u, v) for u in top for v in bottom))
    G.graph['name'] = f'complete_bipartite_graph({n1}, {n2})'
    return G