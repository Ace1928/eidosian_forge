from collections import defaultdict, deque
from itertools import chain, combinations, islice
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@nx._dispatch
def find_cliques(G, nodes=None):
    """Returns all maximal cliques in an undirected graph.

    For each node *n*, a *maximal clique for n* is a largest complete
    subgraph containing *n*. The largest maximal clique is sometimes
    called the *maximum clique*.

    This function returns an iterator over cliques, each of which is a
    list of nodes. It is an iterative implementation, so should not
    suffer from recursion depth issues.

    This function accepts a list of `nodes` and only the maximal cliques
    containing all of these `nodes` are returned. It can considerably speed up
    the running time if some specific cliques are desired.

    Parameters
    ----------
    G : NetworkX graph
        An undirected graph.

    nodes : list, optional (default=None)
        If provided, only yield *maximal cliques* containing all nodes in `nodes`.
        If `nodes` isn't a clique itself, a ValueError is raised.

    Returns
    -------
    iterator
        An iterator over maximal cliques, each of which is a list of
        nodes in `G`. If `nodes` is provided, only the maximal cliques
        containing all the nodes in `nodes` are returned. The order of
        cliques is arbitrary.

    Raises
    ------
    ValueError
        If `nodes` is not a clique.

    Examples
    --------
    >>> from pprint import pprint  # For nice dict formatting
    >>> G = nx.karate_club_graph()
    >>> sum(1 for c in nx.find_cliques(G))  # The number of maximal cliques in G
    36
    >>> max(nx.find_cliques(G), key=len)  # The largest maximal clique in G
    [0, 1, 2, 3, 13]

    The size of the largest maximal clique is known as the *clique number* of
    the graph, which can be found directly with:

    >>> max(len(c) for c in nx.find_cliques(G))
    5

    One can also compute the number of maximal cliques in `G` that contain a given
    node. The following produces a dictionary keyed by node whose
    values are the number of maximal cliques in `G` that contain the node:

    >>> pprint({n: sum(1 for c in nx.find_cliques(G) if n in c) for n in G})
    {0: 13,
     1: 6,
     2: 7,
     3: 3,
     4: 2,
     5: 3,
     6: 3,
     7: 1,
     8: 3,
     9: 2,
     10: 2,
     11: 1,
     12: 1,
     13: 2,
     14: 1,
     15: 1,
     16: 1,
     17: 1,
     18: 1,
     19: 2,
     20: 1,
     21: 1,
     22: 1,
     23: 3,
     24: 2,
     25: 2,
     26: 1,
     27: 3,
     28: 2,
     29: 2,
     30: 2,
     31: 4,
     32: 9,
     33: 14}

    Or, similarly, the maximal cliques in `G` that contain a given node.
    For example, the 4 maximal cliques that contain node 31:

    >>> [c for c in nx.find_cliques(G) if 31 in c]
    [[0, 31], [33, 32, 31], [33, 28, 31], [24, 25, 31]]

    See Also
    --------
    find_cliques_recursive
        A recursive version of the same algorithm.

    Notes
    -----
    To obtain a list of all maximal cliques, use
    `list(find_cliques(G))`. However, be aware that in the worst-case,
    the length of this list can be exponential in the number of nodes in
    the graph. This function avoids storing all cliques in memory by
    only keeping current candidate node lists in memory during its search.

    This implementation is based on the algorithm published by Bron and
    Kerbosch (1973) [1]_, as adapted by Tomita, Tanaka and Takahashi
    (2006) [2]_ and discussed in Cazals and Karande (2008) [3]_. It
    essentially unrolls the recursion used in the references to avoid
    issues of recursion stack depth (for a recursive implementation, see
    :func:`find_cliques_recursive`).

    This algorithm ignores self-loops and parallel edges, since cliques
    are not conventionally defined with such edges.

    References
    ----------
    .. [1] Bron, C. and Kerbosch, J.
       "Algorithm 457: finding all cliques of an undirected graph".
       *Communications of the ACM* 16, 9 (Sep. 1973), 575--577.
       <http://portal.acm.org/citation.cfm?doid=362342.362367>

    .. [2] Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi,
       "The worst-case time complexity for generating all maximal
       cliques and computational experiments",
       *Theoretical Computer Science*, Volume 363, Issue 1,
       Computing and Combinatorics,
       10th Annual International Conference on
       Computing and Combinatorics (COCOON 2004), 25 October 2006, Pages 28--42
       <https://doi.org/10.1016/j.tcs.2006.06.015>

    .. [3] F. Cazals, C. Karande,
       "A note on the problem of reporting maximal cliques",
       *Theoretical Computer Science*,
       Volume 407, Issues 1--3, 6 November 2008, Pages 564--568,
       <https://doi.org/10.1016/j.tcs.2008.05.010>

    """
    if len(G) == 0:
        return
    adj = {u: {v for v in G[u] if v != u} for u in G}
    Q = nodes[:] if nodes is not None else []
    cand = set(G)
    for node in Q:
        if node not in cand:
            raise ValueError(f'The given `nodes` {nodes} do not form a clique')
        cand &= adj[node]
    if not cand:
        yield Q[:]
        return
    subg = cand.copy()
    stack = []
    Q.append(None)
    u = max(subg, key=lambda u: len(cand & adj[u]))
    ext_u = cand - adj[u]
    try:
        while True:
            if ext_u:
                q = ext_u.pop()
                cand.remove(q)
                Q[-1] = q
                adj_q = adj[q]
                subg_q = subg & adj_q
                if not subg_q:
                    yield Q[:]
                else:
                    cand_q = cand & adj_q
                    if cand_q:
                        stack.append((subg, cand, ext_u))
                        Q.append(None)
                        subg = subg_q
                        cand = cand_q
                        u = max(subg, key=lambda u: len(cand & adj[u]))
                        ext_u = cand - adj[u]
            else:
                Q.pop()
                subg, cand, ext_u = stack.pop()
    except IndexError:
        pass