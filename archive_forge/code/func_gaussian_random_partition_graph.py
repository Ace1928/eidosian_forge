import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(6)
@nx._dispatch(graphs=None)
def gaussian_random_partition_graph(n, s, v, p_in, p_out, directed=False, seed=None):
    """Generate a Gaussian random partition graph.

    A Gaussian random partition graph is created by creating k partitions
    each with a size drawn from a normal distribution with mean s and variance
    s/v. Nodes are connected within clusters with probability p_in and
    between clusters with probability p_out[1]

    Parameters
    ----------
    n : int
      Number of nodes in the graph
    s : float
      Mean cluster size
    v : float
      Shape parameter. The variance of cluster size distribution is s/v.
    p_in : float
      Probability of intra cluster connection.
    p_out : float
      Probability of inter cluster connection.
    directed : boolean, optional default=False
      Whether to create a directed graph or not
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : NetworkX Graph or DiGraph
      gaussian random partition graph

    Raises
    ------
    NetworkXError
      If s is > n
      If p_in or p_out is not in [0,1]

    Notes
    -----
    Note the number of partitions is dependent on s,v and n, and that the
    last partition may be considerably smaller, as it is sized to simply
    fill out the nodes [1]

    See Also
    --------
    random_partition_graph

    Examples
    --------
    >>> G = nx.gaussian_random_partition_graph(100, 10, 10, 0.25, 0.1)
    >>> len(G)
    100

    References
    ----------
    .. [1] Ulrik Brandes, Marco Gaertler, Dorothea Wagner,
       Experiments on Graph Clustering Algorithms,
       In the proceedings of the 11th Europ. Symp. Algorithms, 2003.
    """
    if s > n:
        raise nx.NetworkXError('s must be <= n')
    assigned = 0
    sizes = []
    while True:
        size = int(seed.gauss(s, s / v + 0.5))
        if size < 1:
            continue
        if assigned + size >= n:
            sizes.append(n - assigned)
            break
        assigned += size
        sizes.append(size)
    return random_partition_graph(sizes, p_in, p_out, seed=seed, directed=directed)