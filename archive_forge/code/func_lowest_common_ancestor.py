from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import combinations_with_replacement
import networkx as nx
from networkx.utils import UnionFind, arbitrary_element, not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def lowest_common_ancestor(G, node1, node2, default=None):
    """Compute the lowest common ancestor of the given pair of nodes.

    Parameters
    ----------
    G : NetworkX directed graph

    node1, node2 : nodes in the graph.

    default : object
        Returned if no common ancestor between `node1` and `node2`

    Returns
    -------
    The lowest common ancestor of node1 and node2,
    or default if they have no common ancestors.

    Examples
    --------
    >>> G = nx.DiGraph()
    >>> nx.add_path(G, (0, 1, 2, 3))
    >>> nx.add_path(G, (0, 4, 3))
    >>> nx.lowest_common_ancestor(G, 2, 4)
    0

    See Also
    --------
    all_pairs_lowest_common_ancestor"""
    ans = list(all_pairs_lowest_common_ancestor(G, pairs=[(node1, node2)]))
    if ans:
        assert len(ans) == 1
        return ans[0][1]
    return default