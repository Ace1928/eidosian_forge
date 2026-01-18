import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
@py_random_state(2)
@nx._dispatch
def strategy_random_sequential(G, colors, seed=None):
    """Returns a random permutation of the nodes of ``G`` as a list.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    """
    nodes = list(G)
    seed.shuffle(nodes)
    return nodes