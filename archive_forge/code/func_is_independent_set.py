import networkx as nx
from networkx.algorithms.approximation import (
def is_independent_set(G, nodes):
    """Returns True if and only if `nodes` is a clique in `G`.

    `G` is a NetworkX graph. `nodes` is an iterable of nodes in
    `G`.

    """
    return G.subgraph(nodes).number_of_edges() == 0