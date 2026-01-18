import networkx as nx
from networkx.utils import py_random_state
@py_random_state(3)
@nx._dispatch(graphs=None)
def k_random_intersection_graph(n, m, k, seed=None):
    """Returns a intersection graph with randomly chosen attribute sets for
    each node that are of equal size (k).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    k : float
        Size of attribute set to assign to each node.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph, uniform_random_intersection_graph

    References
    ----------
    .. [1] Godehardt, E., and Jaworski, J.
       Two models of random intersection graphs and their applications.
       Electronic Notes in Discrete Mathematics 10 (2001), 129--132.
    """
    G = nx.empty_graph(n + m)
    mset = range(n, n + m)
    for v in range(n):
        targets = seed.sample(mset, k)
        G.add_edges_from(zip([v] * len(targets), targets))
    return nx.projected_graph(G, range(n))