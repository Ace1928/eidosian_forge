import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
@py_random_state(2)
@nx._dispatch(graphs=None)
def random_powerlaw_tree_sequence(n, gamma=3, seed=None, tries=100):
    """Returns a degree sequence for a tree with a power law distribution.

    Parameters
    ----------
    n : int,
        The number of nodes.
    gamma : float
        Exponent of the power law.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    tries : int
        Number of attempts to adjust the sequence to make it a tree.

    Raises
    ------
    NetworkXError
        If no valid sequence is found within the maximum number of
        attempts.

    Notes
    -----
    A trial power law degree sequence is chosen and then elements are
    swapped with new elements from a power law distribution until
    the sequence makes a tree (by checking, for example, that the number of
    edges is one smaller than the number of nodes).

    """
    z = nx.utils.powerlaw_sequence(n, exponent=gamma, seed=seed)
    zseq = [min(n, max(round(s), 0)) for s in z]
    z = nx.utils.powerlaw_sequence(tries, exponent=gamma, seed=seed)
    swap = [min(n, max(round(s), 0)) for s in z]
    for deg in swap:
        if 2 * n - sum(zseq) == 2:
            return zseq
        index = seed.randint(0, n - 1)
        zseq[index] = swap.pop()
    raise nx.NetworkXError(f'Exceeded max ({tries}) attempts for a valid tree sequence.')