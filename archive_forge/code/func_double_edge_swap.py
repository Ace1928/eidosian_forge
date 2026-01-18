import math
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(3)
@nx._dispatch
def double_edge_swap(G, nswap=1, max_tries=100, seed=None):
    """Swap two edges in the graph while keeping the node degrees fixed.

    A double-edge swap removes two randomly chosen edges u-v and x-y
    and creates the new edges u-x and v-y::

     u--v            u  v
            becomes  |  |
     x--y            x  y

    If either the edge u-x or v-y already exist no swap is performed
    and another attempt is made to find a suitable edge pair.

    Parameters
    ----------
    G : graph
       An undirected graph

    nswap : integer (optional, default=1)
       Number of double-edge swaps to perform

    max_tries : integer (optional)
       Maximum number of attempts to swap edges

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : graph
       The graph after double edge swaps.

    Raises
    ------
    NetworkXError
        If `G` is directed, or
        If `nswap` > `max_tries`, or
        If there are fewer than 4 nodes or 2 edges in `G`.
    NetworkXAlgorithmError
        If the number of swap attempts exceeds `max_tries` before `nswap` swaps are made

    Notes
    -----
    Does not enforce any connectivity constraints.

    The graph G is modified in place.
    """
    if G.is_directed():
        raise nx.NetworkXError('double_edge_swap() not defined for directed graphs. Use directed_edge_swap instead.')
    if nswap > max_tries:
        raise nx.NetworkXError('Number of swaps > number of tries allowed.')
    if len(G) < 4:
        raise nx.NetworkXError('Graph has fewer than four nodes.')
    if len(G.edges) < 2:
        raise nx.NetworkXError('Graph has fewer than 2 edges')
    n = 0
    swapcount = 0
    keys, degrees = zip(*G.degree())
    cdf = nx.utils.cumulative_distribution(degrees)
    discrete_sequence = nx.utils.discrete_sequence
    while swapcount < nswap:
        ui, xi = discrete_sequence(2, cdistribution=cdf, seed=seed)
        if ui == xi:
            continue
        u = keys[ui]
        x = keys[xi]
        v = seed.choice(list(G[u]))
        y = seed.choice(list(G[x]))
        if v == y:
            continue
        if x not in G[u] and y not in G[v]:
            G.add_edge(u, x)
            G.add_edge(v, y)
            G.remove_edge(u, v)
            G.remove_edge(x, y)
            swapcount += 1
        if n >= max_tries:
            e = f'Maximum number of swap attempts ({n}) exceeded before desired swaps achieved ({nswap}).'
            raise nx.NetworkXAlgorithmError(e)
        n += 1
    return G