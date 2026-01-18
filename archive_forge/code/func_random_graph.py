import math
import numbers
from functools import reduce
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
@py_random_state(3)
@nx._dispatch(graphs=None)
def random_graph(n, m, p, seed=None, directed=False):
    """Returns a bipartite random graph.

    This is a bipartite version of the binomial (Erdős-Rényi) graph.
    The graph is composed of two partitions. Set A has nodes 0 to
    (n - 1) and set B has nodes n to (n + m - 1).

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set.
    m : int
        The number of nodes in the second bipartite set.
    p : float
        Probability for edge creation.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    directed : bool, optional (default=False)
        If True return a directed graph

    Notes
    -----
    The bipartite random graph algorithm chooses each of the n*m (undirected)
    or 2*nm (directed) possible edges with probability p.

    This algorithm is $O(n+m)$ where $m$ is the expected number of edges.

    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.random_graph

    See Also
    --------
    gnp_random_graph, configuration_model

    References
    ----------
    .. [1] Vladimir Batagelj and Ulrik Brandes,
       "Efficient generation of large random networks",
       Phys. Rev. E, 71, 036113, 2005.
    """
    G = nx.Graph()
    G = _add_nodes_with_bipartite_label(G, n, m)
    if directed:
        G = nx.DiGraph(G)
    G.name = f'fast_gnp_random_graph({n},{m},{p})'
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_bipartite_graph(n, m)
    lp = math.log(1.0 - p)
    v = 0
    w = -1
    while v < n:
        lr = math.log(1.0 - seed.random())
        w = w + 1 + int(lr / lp)
        while w >= m and v < n:
            w = w - m
            v = v + 1
        if v < n:
            G.add_edge(v, n + w)
    if directed:
        v = 0
        w = -1
        while v < n:
            lr = math.log(1.0 - seed.random())
            w = w + 1 + int(lr / lp)
            while w >= m and v < n:
                w = w - m
                v = v + 1
            if v < n:
                G.add_edge(n + w, v)
    return G