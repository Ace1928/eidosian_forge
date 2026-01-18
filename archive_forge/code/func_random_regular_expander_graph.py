import itertools
import networkx as nx
@nx.utils.decorators.np_random_state('seed')
@nx._dispatchable(graphs=None, returns_graph=True)
def random_regular_expander_graph(n, d, *, epsilon=0, create_using=None, max_tries=100, seed=None):
    """Returns a random regular expander graph on $n$ nodes with degree $d$.

    An expander graph is a sparse graph with strong connectivity properties. [1]_

    More precisely the returned graph is a $(n, d, \\lambda)$-expander with
    $\\lambda = 2 \\sqrt{d - 1} + \\epsilon$, close to the Alon-Boppana bound. [2]_

    In the case where $\\epsilon = 0$ it returns a Ramanujan graph.
    A Ramanujan graph has spectral gap almost as large as possible,
    which makes them excellent expanders. [3]_

    Parameters
    ----------
    n : int
      The number of nodes.
    d : int
      The degree of each node.
    epsilon : int, float, default=0
    max_tries : int, (default: 100)
      The number of allowed loops, also used in the maybe_regular_expander utility
    seed : (default: None)
      Seed used to set random number generation state. See :ref`Randomness<randomness>`.

    Raises
    ------
    NetworkXError
        If max_tries is reached

    Examples
    --------
    >>> G = nx.random_regular_expander_graph(20, 4)
    >>> nx.is_regular_expander(G)
    True

    Notes
    -----
    This loops over `maybe_regular_expander` and can be slow when
    $n$ is too big or $\\epsilon$ too small.

    See Also
    --------
    maybe_regular_expander
    is_regular_expander

    References
    ----------
    .. [1] Expander graph, https://en.wikipedia.org/wiki/Expander_graph
    .. [2] Alon-Boppana bound, https://en.wikipedia.org/wiki/Alon%E2%80%93Boppana_bound
    .. [3] Ramanujan graphs, https://en.wikipedia.org/wiki/Ramanujan_graph

    """
    G = maybe_regular_expander(n, d, create_using=create_using, max_tries=max_tries, seed=seed)
    iterations = max_tries
    while not is_regular_expander(G, epsilon=epsilon):
        iterations -= 1
        G = maybe_regular_expander(n=n, d=d, create_using=create_using, max_tries=max_tries, seed=seed)
        if iterations == 0:
            raise nx.NetworkXError('Too many iterations in random_regular_expander_graph')
    return G