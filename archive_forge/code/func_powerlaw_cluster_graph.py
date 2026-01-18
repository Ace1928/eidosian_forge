import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
@py_random_state(3)
@nx._dispatch(graphs=None)
def powerlaw_cluster_graph(n, m, p, seed=None):
    """Holme and Kim algorithm for growing graphs with powerlaw
    degree distribution and approximate average clustering.

    Parameters
    ----------
    n : int
        the number of nodes
    m : int
        the number of random edges to add for each new node
    p : float,
        Probability of adding a triangle after adding a random edge
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    The average clustering has a hard time getting above a certain
    cutoff that depends on `m`.  This cutoff is often quite low.  The
    transitivity (fraction of triangles to possible triangles) seems to
    decrease with network size.

    It is essentially the Barabási–Albert (BA) growth model with an
    extra step that each random edge is followed by a chance of
    making an edge to one of its neighbors too (and thus a triangle).

    This algorithm improves on BA in the sense that it enables a
    higher average clustering to be attained if desired.

    It seems possible to have a disconnected graph with this algorithm
    since the initial `m` nodes may not be all linked to a new node
    on the first iteration like the BA model.

    Raises
    ------
    NetworkXError
        If `m` does not satisfy ``1 <= m <= n`` or `p` does not
        satisfy ``0 <= p <= 1``.

    References
    ----------
    .. [1] P. Holme and B. J. Kim,
       "Growing scale-free networks with tunable clustering",
       Phys. Rev. E, 65, 026107, 2002.
    """
    if m < 1 or n < m:
        raise nx.NetworkXError(f'NetworkXError must have m>1 and m<n, m={m},n={n}')
    if p > 1 or p < 0:
        raise nx.NetworkXError(f'NetworkXError p must be in [0,1], p={p}')
    G = empty_graph(m)
    repeated_nodes = list(G.nodes())
    source = m
    while source < n:
        possible_targets = _random_subset(repeated_nodes, m, seed)
        target = possible_targets.pop()
        G.add_edge(source, target)
        repeated_nodes.append(target)
        count = 1
        while count < m:
            if seed.random() < p:
                neighborhood = [nbr for nbr in G.neighbors(target) if not G.has_edge(source, nbr) and nbr != source]
                if neighborhood:
                    nbr = seed.choice(neighborhood)
                    G.add_edge(source, nbr)
                    repeated_nodes.append(nbr)
                    count = count + 1
                    continue
            target = possible_targets.pop()
            G.add_edge(source, target)
            repeated_nodes.append(target)
            count = count + 1
        repeated_nodes.extend([source] * m)
        source += 1
    return G