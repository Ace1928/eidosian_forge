import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
@py_random_state('seed')
@nx._dispatch(graphs=None)
def random_unlabeled_rooted_forest(n, *, q=None, number_of_forests=None, seed=None):
    """Returns a forest or list of forests selected at random.

    Returns one or more (depending on `number_of_forests`)
    unlabeled rooted forests with `n` nodes, and with no more than
    `q` nodes per tree, drawn uniformly at random.
    The "roots" graph attribute identifies the roots of the forest.

    Parameters
    ----------
    n : int
        The number of nodes
    q : int or None (default)
        The maximum number of nodes per tree.
    number_of_forests : int or None (default)
        If not None, this number of forests is generated and returned.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph` or list of :class:`networkx.Graph`
        A single `networkx.Graph` (or a list thereof, if `number_of_forests`
        is specified) with nodes in the set {0, …, *n* - 1}.
        The "roots" graph attribute is a set containing the roots
        of the trees in the forest.

    Notes
    -----
    This function implements the algorithm "Forest" of [1]_.
    The algorithm needs to compute some counting functions
    that are relatively expensive: in case several trees are needed,
    it is advisable to use the `number_of_forests` optional argument
    to reuse the counting functions.

    Raises
    ------
    ValueError
        If `n` is non-zero but `q` is zero.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    if q is None:
        q = n
    if q == 0 and n != 0:
        raise ValueError('q must be a positive integer if n is positive.')
    cache_trees = [0, 1]
    cache_forests = [1]
    if number_of_forests is None:
        g, nodes, rs = _random_unlabeled_rooted_forest(n, q, cache_trees, cache_forests, seed)
        return _to_nx(g, nodes, roots=set(rs))
    res = []
    for i in range(number_of_forests):
        g, nodes, rs = _random_unlabeled_rooted_forest(n, q, cache_trees, cache_forests, seed)
        res.append(_to_nx(g, nodes, roots=set(rs)))
    return res