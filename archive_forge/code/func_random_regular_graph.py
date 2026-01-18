import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
@py_random_state(2)
@nx._dispatch(graphs=None)
def random_regular_graph(d, n, seed=None):
    """Returns a random $d$-regular graph on $n$ nodes.

    A regular graph is a graph where each node has the same number of neighbors.

    The resulting graph has no self-loops or parallel edges.

    Parameters
    ----------
    d : int
      The degree of each node.
    n : integer
      The number of nodes. The value of $n \\times d$ must be even.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    The nodes are numbered from $0$ to $n - 1$.

    Kim and Vu's paper [2]_ shows that this algorithm samples in an
    asymptotically uniform way from the space of random graphs when
    $d = O(n^{1 / 3 - \\epsilon})$.

    Raises
    ------

    NetworkXError
        If $n \\times d$ is odd or $d$ is greater than or equal to $n$.

    References
    ----------
    .. [1] A. Steger and N. Wormald,
       Generating random regular graphs quickly,
       Probability and Computing 8 (1999), 377-396, 1999.
       https://doi.org/10.1017/S0963548399003867

    .. [2] Jeong Han Kim and Van H. Vu,
       Generating random regular graphs,
       Proceedings of the thirty-fifth ACM symposium on Theory of computing,
       San Diego, CA, USA, pp 213--222, 2003.
       http://portal.acm.org/citation.cfm?id=780542.780576
    """
    if n * d % 2 != 0:
        raise nx.NetworkXError('n * d must be even')
    if not 0 <= d < n:
        raise nx.NetworkXError('the 0 <= d < n inequality must be satisfied')
    if d == 0:
        return empty_graph(n)

    def _suitable(edges, potential_edges):
        if not potential_edges:
            return True
        for s1 in potential_edges:
            for s2 in potential_edges:
                if s1 == s2:
                    break
                if s1 > s2:
                    s1, s2 = (s2, s1)
                if (s1, s2) not in edges:
                    return True
        return False

    def _try_creation():
        edges = set()
        stubs = list(range(n)) * d
        while stubs:
            potential_edges = defaultdict(lambda: 0)
            seed.shuffle(stubs)
            stubiter = iter(stubs)
            for s1, s2 in zip(stubiter, stubiter):
                if s1 > s2:
                    s1, s2 = (s2, s1)
                if s1 != s2 and (s1, s2) not in edges:
                    edges.add((s1, s2))
                else:
                    potential_edges[s1] += 1
                    potential_edges[s2] += 1
            if not _suitable(edges, potential_edges):
                return None
            stubs = [node for node, potential in potential_edges.items() for _ in range(potential)]
        return edges
    edges = _try_creation()
    while edges is None:
        edges = _try_creation()
    G = nx.Graph()
    G.add_edges_from(edges)
    return G