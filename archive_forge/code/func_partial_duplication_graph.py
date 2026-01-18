import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import py_random_state
@py_random_state(4)
@nx._dispatch(graphs=None)
def partial_duplication_graph(N, n, p, q, seed=None):
    """Returns a random graph using the partial duplication model.

    Parameters
    ----------
    N : int
        The total number of nodes in the final graph.

    n : int
        The number of nodes in the initial clique.

    p : float
        The probability of joining each neighbor of a node to the
        duplicate node. Must be a number in the between zero and one,
        inclusive.

    q : float
        The probability of joining the source node to the duplicate
        node. Must be a number in the between zero and one, inclusive.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Notes
    -----
    A graph of nodes is grown by creating a fully connected graph
    of size `n`. The following procedure is then repeated until
    a total of `N` nodes have been reached.

    1. A random node, *u*, is picked and a new node, *v*, is created.
    2. For each neighbor of *u* an edge from the neighbor to *v* is created
       with probability `p`.
    3. An edge from *u* to *v* is created with probability `q`.

    This algorithm appears in [1].

    This implementation allows the possibility of generating
    disconnected graphs.

    References
    ----------
    .. [1] Knudsen Michael, and Carsten Wiuf. "A Markov chain approach to
           randomly grown graphs." Journal of Applied Mathematics 2008.
           <https://doi.org/10.1155/2008/190836>

    """
    if p < 0 or p > 1 or q < 0 or (q > 1):
        msg = 'partial duplication graph must have 0 <= p, q <= 1.'
        raise NetworkXError(msg)
    if n > N:
        raise NetworkXError('partial duplication graph must have n <= N.')
    G = nx.complete_graph(n)
    for new_node in range(n, N):
        src_node = seed.randint(0, new_node - 1)
        G.add_node(new_node)
        for neighbor_node in list(nx.all_neighbors(G, src_node)):
            if seed.random() < p:
                G.add_edge(new_node, neighbor_node)
        if seed.random() < q:
            G.add_edge(new_node, src_node)
    return G