import itertools
from collections import defaultdict, deque
import networkx as nx
from networkx.utils import arbitrary_element, py_random_state
@nx._dispatch
def strategy_connected_sequential_bfs(G, colors):
    """Returns an iterable over nodes in ``G`` in the order given by a
    breadth-first traversal.

    The generated sequence has the property that for each node except
    the first, at least one neighbor appeared earlier in the sequence.

    ``G`` is a NetworkX graph. ``colors`` is ignored.

    """
    return strategy_connected_sequential(G, colors, 'bfs')