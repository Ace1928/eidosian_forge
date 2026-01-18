import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
@py_random_state('seed')
@nx._dispatch(graphs=None)
def random_labeled_rooted_tree(n, *, seed=None):
    """Returns a labeled rooted tree with `n` nodes.

    The returned tree is chosen uniformly at random from all labeled rooted trees.

    Parameters
    ----------
    n : int
        The number of nodes
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph`
        A `networkx.Graph` with integer nodes 0 <= node <= `n` - 1.
        The root of the tree is selected uniformly from the nodes.
        The "root" graph attribute identifies the root of the tree.

    Notes
    -----
    This function returns the result of :func:`random_labeled_tree`
    with a randomly selected root.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).
    """
    t = random_labeled_tree(n, seed=seed)
    t.graph['root'] = seed.randint(0, n - 1)
    return t