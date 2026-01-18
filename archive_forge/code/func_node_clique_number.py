from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def node_clique_number(G, nodes=None, cliques=None, separate_nodes=False):
    """Returns the size of the largest maximal clique containing each given node.

    Returns a single or list depending on input nodes.
    An optional list of cliques can be input if already computed.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    cliques : list, optional (default=None)
        A list of cliques, each of which is itself a list of nodes.
        If not specified, the list of all cliques will be computed
        using :func:`find_cliques`.

    Returns
    -------
    int or dict
        If `nodes` is a single node, returns the size of the
        largest maximal clique in `G` containing that node.
        Otherwise return a dict keyed by node to the size
        of the largest maximal clique containing that node.

    See Also
    --------
    find_cliques
        find_cliques yields the maximal cliques of G.
        It accepts a `nodes` argument which restricts consideration to
        maximal cliques containing all the given `nodes`.
        The search for the cliques is optimized for `nodes`.
    """
    if cliques is None:
        if nodes is not None:
            if nodes in G:
                return max((len(c) for c in find_cliques(nx.ego_graph(G, nodes))))
            return {n: max((len(c) for c in find_cliques(nx.ego_graph(G, n)))) for n in nodes}
        cliques = list(find_cliques(G))
    if nodes in G:
        return max((len(c) for c in cliques if nodes in c))
    size_for_n = defaultdict(int)
    for c in cliques:
        size_of_c = len(c)
        for n in c:
            if size_for_n[n] < size_of_c:
                size_for_n[n] = size_of_c
    if nodes is None:
        return size_for_n
    return {n: size_for_n[n] for n in nodes}