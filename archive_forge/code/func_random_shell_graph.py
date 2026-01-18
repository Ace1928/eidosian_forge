import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
@py_random_state(1)
@nx._dispatch(graphs=None)
def random_shell_graph(constructor, seed=None):
    """Returns a random shell graph for the constructor given.

    Parameters
    ----------
    constructor : list of three-tuples
        Represents the parameters for a shell, starting at the center
        shell.  Each element of the list must be of the form `(n, m,
        d)`, where `n` is the number of nodes in the shell, `m` is
        the number of edges in the shell, and `d` is the ratio of
        inter-shell (next) edges to intra-shell edges. If `d` is zero,
        there will be no intra-shell edges, and if `d` is one there
        will be all possible intra-shell edges.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Examples
    --------
    >>> constructor = [(10, 20, 0.8), (20, 40, 0.8)]
    >>> G = nx.random_shell_graph(constructor)

    """
    G = empty_graph(0)
    glist = []
    intra_edges = []
    nnodes = 0
    for n, m, d in constructor:
        inter_edges = int(m * d)
        intra_edges.append(m - inter_edges)
        g = nx.convert_node_labels_to_integers(gnm_random_graph(n, inter_edges, seed=seed), first_label=nnodes)
        glist.append(g)
        nnodes += n
        G = nx.operators.union(G, g)
    for gi in range(len(glist) - 1):
        nlist1 = list(glist[gi])
        nlist2 = list(glist[gi + 1])
        total_edges = intra_edges[gi]
        edge_count = 0
        while edge_count < total_edges:
            u = seed.choice(nlist1)
            v = seed.choice(nlist2)
            if u == v or G.has_edge(u, v):
                continue
            else:
                G.add_edge(u, v)
                edge_count = edge_count + 1
    return G