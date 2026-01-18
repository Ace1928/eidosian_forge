import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
@py_random_state(3)
@nx._dispatch(graphs=None)
def newman_watts_strogatz_graph(n, k, p, seed=None):
    """Returns a Newman–Watts–Strogatz small-world graph.

    Parameters
    ----------
    n : int
        The number of nodes.
    k : int
        Each node is joined with its `k` nearest neighbors in a ring
        topology.
    p : float
        The probability of adding a new edge for each edge.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    First create a ring over $n$ nodes [1]_.  Then each node in the ring is
    connected with its $k$ nearest neighbors (or $k - 1$ neighbors if $k$
    is odd).  Then shortcuts are created by adding new edges as follows: for
    each edge $(u, v)$ in the underlying "$n$-ring with $k$ nearest
    neighbors" with probability $p$ add a new edge $(u, w)$ with
    randomly-chosen existing node $w$.  In contrast with
    :func:`watts_strogatz_graph`, no edges are removed.

    See Also
    --------
    watts_strogatz_graph

    References
    ----------
    .. [1] M. E. J. Newman and D. J. Watts,
       Renormalization group analysis of the small-world network model,
       Physics Letters A, 263, 341, 1999.
       https://doi.org/10.1016/S0375-9601(99)00757-4
    """
    if k > n:
        raise nx.NetworkXError('k>=n, choose smaller k or larger n')
    if k == n:
        return nx.complete_graph(n)
    G = empty_graph(n)
    nlist = list(G.nodes())
    fromv = nlist
    for j in range(1, k // 2 + 1):
        tov = fromv[j:] + fromv[0:j]
        for i in range(len(fromv)):
            G.add_edge(fromv[i], tov[i])
    e = list(G.edges())
    for u, v in e:
        if seed.random() < p:
            w = seed.choice(nlist)
            while w == u or G.has_edge(u, w):
                w = seed.choice(nlist)
                if G.degree(u) >= n - 1:
                    break
            else:
                G.add_edge(u, w)
    return G