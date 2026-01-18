from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import combinations_with_replacement
import networkx as nx
from networkx.utils import UnionFind, arbitrary_element, not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def all_pairs_lowest_common_ancestor(G, pairs=None):
    """Return the lowest common ancestor of all pairs or the provided pairs

    Parameters
    ----------
    G : NetworkX directed graph

    pairs : iterable of pairs of nodes, optional (default: all pairs)
        The pairs of nodes of interest.
        If None, will find the LCA of all pairs of nodes.

    Yields
    ------
    ((node1, node2), lca) : 2-tuple
        Where lca is least common ancestor of node1 and node2.
        Note that for the default case, the order of the node pair is not considered,
        e.g. you will not get both ``(a, b)`` and ``(b, a)``

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is null.
    NetworkXError
        If `G` is not a DAG.

    Examples
    --------
    The default behavior is to yield the lowest common ancestor for all
    possible combinations of nodes in `G`, including self-pairings:

    >>> G = nx.DiGraph([(0, 1), (0, 3), (1, 2)])
    >>> dict(nx.all_pairs_lowest_common_ancestor(G))
    {(0, 0): 0, (0, 1): 0, (0, 3): 0, (0, 2): 0, (1, 1): 1, (1, 3): 0, (1, 2): 1, (3, 3): 3, (3, 2): 0, (2, 2): 2}

    The pairs argument can be used to limit the output to only the
    specified node pairings:

    >>> dict(nx.all_pairs_lowest_common_ancestor(G, pairs=[(1, 2), (2, 3)]))
    {(1, 2): 1, (2, 3): 0}

    Notes
    -----
    Only defined on non-null directed acyclic graphs.

    See Also
    --------
    lowest_common_ancestor
    """
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError('LCA only defined on directed acyclic graphs.')
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('LCA meaningless on null graphs.')
    if pairs is None:
        pairs = combinations_with_replacement(G, 2)
    else:
        pairs = dict.fromkeys(pairs)
        nodeset = set(G)
        for pair in pairs:
            if set(pair) - nodeset:
                raise nx.NodeNotFound(f'Node(s) {set(pair) - nodeset} from pair {pair} not in G.')

    def generate_lca_from_pairs(G, pairs):
        ancestor_cache = {}
        for v, w in pairs:
            if v not in ancestor_cache:
                ancestor_cache[v] = nx.ancestors(G, v)
                ancestor_cache[v].add(v)
            if w not in ancestor_cache:
                ancestor_cache[w] = nx.ancestors(G, w)
                ancestor_cache[w].add(w)
            common_ancestors = ancestor_cache[v] & ancestor_cache[w]
            if common_ancestors:
                common_ancestor = next(iter(common_ancestors))
                while True:
                    successor = None
                    for lower_ancestor in G.successors(common_ancestor):
                        if lower_ancestor in common_ancestors:
                            successor = lower_ancestor
                            break
                    if successor is None:
                        break
                    common_ancestor = successor
                yield ((v, w), common_ancestor)
    return generate_lca_from_pairs(G, pairs)