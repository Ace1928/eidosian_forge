import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
@py_random_state('seed')
@nx._dispatch(graphs=None)
def random_unlabeled_rooted_tree(n, *, number_of_trees=None, seed=None):
    """Returns a number of unlabeled rooted trees uniformly at random

    Returns one or more (depending on `number_of_trees`)
    unlabeled rooted trees with `n` nodes drawn uniformly
    at random.

    Parameters
    ----------
    n : int
        The number of nodes
    number_of_trees : int or None (default)
        If not None, this number of trees is generated and returned.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Returns
    -------
    :class:`networkx.Graph` or list of :class:`networkx.Graph`
        A single `networkx.Graph` (or a list thereof, if `number_of_trees`
        is specified) with nodes in the set {0, …, *n* - 1}.
        The "root" graph attribute identifies the root of the tree.

    Notes
    -----
    The trees are generated using the "RANRUT" algorithm from [1]_.
    The algorithm needs to compute some counting functions
    that are relatively expensive: in case several trees are needed,
    it is advisable to use the `number_of_trees` optional argument
    to reuse the counting functions.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).

    References
    ----------
    .. [1] Nijenhuis, Albert, and Wilf, Herbert S.
        "Combinatorial algorithms: for computers and calculators."
        Academic Press, 1978.
        https://doi.org/10.1016/C2013-0-11243-3
    """
    if n == 0:
        raise nx.NetworkXPointlessConcept('the null graph is not a tree')
    cache_trees = [0, 1]
    if number_of_trees is None:
        return _to_nx(*_random_unlabeled_rooted_tree(n, cache_trees, seed), root=0)
    return [_to_nx(*_random_unlabeled_rooted_tree(n, cache_trees, seed), root=0) for i in range(number_of_trees)]