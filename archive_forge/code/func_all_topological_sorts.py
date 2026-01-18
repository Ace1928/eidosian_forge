import heapq
from collections import deque
from functools import partial
from itertools import chain, combinations, product, starmap
from math import gcd
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for, pairwise
@not_implemented_for('undirected')
@nx._dispatch
def all_topological_sorts(G):
    """Returns a generator of _all_ topological sorts of the directed graph G.

    A topological sort is a nonunique permutation of the nodes such that an
    edge from u to v implies that u appears before v in the topological sort
    order.

    Parameters
    ----------
    G : NetworkX DiGraph
        A directed graph

    Yields
    ------
    topological_sort_order : list
        a list of nodes in `G`, representing one of the topological sort orders

    Raises
    ------
    NetworkXNotImplemented
        If `G` is not directed
    NetworkXUnfeasible
        If `G` is not acyclic

    Examples
    --------
    To enumerate all topological sorts of directed graph:

    >>> DG = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
    >>> list(nx.all_topological_sorts(DG))
    [[1, 2, 4, 3], [1, 2, 3, 4]]

    Notes
    -----
    Implements an iterative version of the algorithm given in [1].

    References
    ----------
    .. [1] Knuth, Donald E., Szwarcfiter, Jayme L. (1974).
       "A Structured Program to Generate All Topological Sorting Arrangements"
       Information Processing Letters, Volume 2, Issue 6, 1974, Pages 153-157,
       ISSN 0020-0190,
       https://doi.org/10.1016/0020-0190(74)90001-5.
       Elsevier (North-Holland), Amsterdam
    """
    if not G.is_directed():
        raise nx.NetworkXError('Topological sort not defined on undirected graphs.')
    count = dict(G.in_degree())
    D = deque([v for v, d in G.in_degree() if d == 0])
    bases = []
    current_sort = []
    while True:
        assert all((count[v] == 0 for v in D))
        if len(current_sort) == len(G):
            yield list(current_sort)
            while len(current_sort) > 0:
                assert len(bases) == len(current_sort)
                q = current_sort.pop()
                for _, j in G.out_edges(q):
                    count[j] += 1
                    assert count[j] >= 0
                while len(D) > 0 and count[D[-1]] > 0:
                    D.pop()
                D.appendleft(q)
                if D[-1] == bases[-1]:
                    bases.pop()
                else:
                    break
        else:
            if len(D) == 0:
                raise nx.NetworkXUnfeasible('Graph contains a cycle.')
            q = D.pop()
            for _, j in G.out_edges(q):
                count[j] -= 1
                assert count[j] >= 0
                if count[j] == 0:
                    D.append(j)
            current_sort.append(q)
            if len(bases) < len(current_sort):
                bases.append(q)
        if len(bases) == 0:
            break