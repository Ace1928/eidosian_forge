import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@nx._dispatch
def ancestors(G, source):
    """Returns all nodes having a path to `source` in `G`.

    Parameters
    ----------
    G : NetworkX Graph
    source : node in `G`

    Returns
    -------
    set()
        The ancestors of `source` in `G`

    Raises
    ------
    NetworkXError
        If node `source` is not in `G`.

    Examples
    --------
    >>> DG = nx.path_graph(5, create_using=nx.DiGraph)
    >>> sorted(nx.ancestors(DG, 2))
    [0, 1]

    The `source` node is not an ancestor of itself, but can be included manually:

    >>> sorted(nx.ancestors(DG, 2) | {2})
    [0, 1, 2]

    See also
    --------
    descendants
    """
    return {child for parent, child in nx.bfs_edges(G, source, reverse=True)}