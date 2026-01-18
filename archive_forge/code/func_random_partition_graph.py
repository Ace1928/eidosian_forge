import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(3)
@nx._dispatch(graphs=None)
def random_partition_graph(sizes, p_in, p_out, seed=None, directed=False):
    """Returns the random partition graph with a partition of sizes.

    A partition graph is a graph of communities with sizes defined by
    s in sizes. Nodes in the same group are connected with probability
    p_in and nodes of different groups are connected with probability
    p_out.

    Parameters
    ----------
    sizes : list of ints
      Sizes of groups
    p_in : float
      probability of edges with in groups
    p_out : float
      probability of edges between groups
    directed : boolean optional, default=False
      Whether to create a directed graph
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : NetworkX Graph or DiGraph
      random partition graph of size sum(gs)

    Raises
    ------
    NetworkXError
      If p_in or p_out is not in [0,1]

    Examples
    --------
    >>> G = nx.random_partition_graph([10, 10, 10], 0.25, 0.01)
    >>> len(G)
    30
    >>> partition = G.graph["partition"]
    >>> len(partition)
    3

    Notes
    -----
    This is a generalization of the planted-l-partition described in
    [1]_.  It allows for the creation of groups of any size.

    The partition is store as a graph attribute 'partition'.

    References
    ----------
    .. [1] Santo Fortunato 'Community Detection in Graphs' Physical Reports
       Volume 486, Issue 3-5 p. 75-174. https://arxiv.org/abs/0906.0612
    """
    if not 0.0 <= p_in <= 1.0:
        raise nx.NetworkXError('p_in must be in [0,1]')
    if not 0.0 <= p_out <= 1.0:
        raise nx.NetworkXError('p_out must be in [0,1]')
    num_blocks = len(sizes)
    p = [[p_out for s in range(num_blocks)] for r in range(num_blocks)]
    for r in range(num_blocks):
        p[r][r] = p_in
    return stochastic_block_model(sizes, p, nodelist=None, seed=seed, directed=directed, selfloops=False, sparse=True)