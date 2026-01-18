import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
@py_random_state('seed')
@nx._dispatch(graphs=None)
def random_labeled_tree(n, *, seed=None):
    """Returns a labeled tree on `n` nodes chosen uniformly at random.

    Generating uniformly distributed random Prüfer sequences and
    converting them into the corresponding trees is a straightforward
    method of generating uniformly distributed random labeled trees.
    This function implements this method.

    Parameters
    ----------
    n : int
        The number of nodes, greater than zero.
    seed : random_state
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`

    Returns
    -------
     :class:`networkx.Graph`
        A `networkx.Graph` with nodes in the set {0, …, *n* - 1}.

    Raises
    ------
    NetworkXPointlessConcept
        If `n` is zero (because the null graph is not a tree).
    """
    if n == 0:
        raise nx.NetworkXPointlessConcept('the null graph is not a tree')
    if n == 1:
        return nx.empty_graph(1)
    return nx.from_prufer_sequence([seed.choice(range(n)) for i in range(n - 2)])