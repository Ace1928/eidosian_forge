import itertools
import math
import networkx as nx
from networkx.utils import py_random_state
@py_random_state(3)
@nx._dispatch(graphs=None)
def relaxed_caveman_graph(l, k, p, seed=None):
    """Returns a relaxed caveman graph.

    A relaxed caveman graph starts with `l` cliques of size `k`.  Edges are
    then randomly rewired with probability `p` to link different cliques.

    Parameters
    ----------
    l : int
      Number of groups
    k : int
      Size of cliques
    p : float
      Probability of rewiring each edge.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    G : NetworkX Graph
      Relaxed Caveman Graph

    Raises
    ------
    NetworkXError
     If p is not in [0,1]

    Examples
    --------
    >>> G = nx.relaxed_caveman_graph(2, 3, 0.1, seed=42)

    References
    ----------
    .. [1] Santo Fortunato, Community Detection in Graphs,
       Physics Reports Volume 486, Issues 3-5, February 2010, Pages 75-174.
       https://arxiv.org/abs/0906.0612
    """
    G = nx.caveman_graph(l, k)
    nodes = list(G)
    for u, v in G.edges():
        if seed.random() < p:
            x = seed.choice(nodes)
            if G.has_edge(u, x):
                continue
            G.remove_edge(u, v)
            G.add_edge(u, x)
    return G