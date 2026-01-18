import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('multigraph')
@not_implemented_for('directed')
@py_random_state(4)
@nx._dispatch
def greedy_k_edge_augmentation(G, k, avail=None, weight=None, seed=None):
    """Greedy algorithm for finding a k-edge-augmentation

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    k : integer
        Desired edge connectivity

    avail : dict or a set of 2 or 3 tuples
        For more details, see :func:`k_edge_augmentation`.

    weight : string
        key to use to find weights if ``avail`` is a set of 3-tuples.
        For more details, see :func:`k_edge_augmentation`.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Yields
    ------
    edge : tuple
        Edges in the greedy augmentation of G

    Notes
    -----
    The algorithm is simple. Edges are incrementally added between parts of the
    graph that are not yet locally k-edge-connected. Then edges are from the
    augmenting set are pruned as long as local-edge-connectivity is not broken.

    This algorithm is greedy and does not provide optimality guarantees. It
    exists only to provide :func:`k_edge_augmentation` with the ability to
    generate a feasible solution for arbitrary k.

    See Also
    --------
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> sorted(greedy_k_edge_augmentation(G, k=2))
    [(1, 7)]
    >>> sorted(greedy_k_edge_augmentation(G, k=1, avail=[]))
    []
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> avail = {(u, v): 1 for (u, v) in complement_edges(G)}
    >>> # randomized pruning process can produce different solutions
    >>> sorted(greedy_k_edge_augmentation(G, k=4, avail=avail, seed=2))
    [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 4), (2, 6), (3, 7), (5, 7)]
    >>> sorted(greedy_k_edge_augmentation(G, k=4, avail=avail, seed=3))
    [(1, 3), (1, 5), (1, 6), (2, 4), (2, 6), (3, 7), (4, 7), (5, 7)]
    """
    aug_edges = []
    done = is_k_edge_connected(G, k)
    if done:
        return
    if avail is None:
        avail_uv = list(complement_edges(G))
        avail_w = [1] * len(avail_uv)
    else:
        avail_uv, avail_w = _unpack_available_edges(avail, weight=weight, G=G)
    tiebreaker = [sum(map(G.degree, uv)) for uv in avail_uv]
    avail_wduv = sorted(zip(avail_w, tiebreaker, avail_uv))
    avail_uv = [uv for w, d, uv in avail_wduv]
    H = G.copy()
    for u, v in avail_uv:
        done = False
        if not is_locally_k_edge_connected(H, u, v, k=k):
            aug_edges.append((u, v))
            H.add_edge(u, v)
            if H.degree(u) >= k and H.degree(v) >= k:
                done = is_k_edge_connected(H, k)
        if done:
            break
    if not done:
        raise nx.NetworkXUnfeasible('not able to k-edge-connect with available edges')
    _compat_shuffle(seed, aug_edges)
    for u, v in list(aug_edges):
        if H.degree(u) <= k or H.degree(v) <= k:
            continue
        H.remove_edge(u, v)
        aug_edges.remove((u, v))
        if not is_k_edge_connected(H, k=k):
            H.add_edge(u, v)
            aug_edges.append((u, v))
    yield from aug_edges