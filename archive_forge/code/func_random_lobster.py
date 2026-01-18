import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
@py_random_state(3)
@nx._dispatch(graphs=None)
def random_lobster(n, p1, p2, seed=None):
    """Returns a random lobster graph.

    A lobster is a tree that reduces to a caterpillar when pruning all
    leaf nodes. A caterpillar is a tree that reduces to a path graph
    when pruning all leaf nodes; setting `p2` to zero produces a caterpillar.

    This implementation iterates on the probabilities `p1` and `p2` to add
    edges at levels 1 and 2, respectively. Graphs are therefore constructed
    iteratively with uniform randomness at each level rather than being selected
    uniformly at random from the set of all possible lobsters.

    Parameters
    ----------
    n : int
        The expected number of nodes in the backbone
    p1 : float
        Probability of adding an edge to the backbone
    p2 : float
        Probability of adding an edge one level beyond backbone
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    Raises
    ------
    NetworkXError
        If `p1` or `p2` parameters are >= 1 because the while loops would never finish.
    """
    p1, p2 = (abs(p1), abs(p2))
    if any((p >= 1 for p in [p1, p2])):
        raise nx.NetworkXError('Probability values for `p1` and `p2` must both be < 1.')
    llen = int(2 * seed.random() * n + 0.5)
    L = path_graph(llen)
    current_node = llen - 1
    for n in range(llen):
        while seed.random() < p1:
            current_node += 1
            L.add_edge(n, current_node)
            cat_node = current_node
            while seed.random() < p2:
                current_node += 1
                L.add_edge(cat_node, current_node)
    return L