from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
@nx._dispatch
def chordless_cycles(G, length_bound=None):
    """Find simple chordless cycles of a graph.

    A `simple cycle` is a closed path where no node appears twice.  In a simple
    cycle, a `chord` is an additional edge between two nodes in the cycle.  A
    `chordless cycle` is a simple cycle without chords.  Said differently, a
    chordless cycle is a cycle C in a graph G where the number of edges in the
    induced graph G[C] is equal to the length of `C`.

    Note that some care must be taken in the case that G is not a simple graph
    nor a simple digraph.  Some authors limit the definition of chordless cycles
    to have a prescribed minimum length; we do not.

        1. We interpret self-loops to be chordless cycles, except in multigraphs
           with multiple loops in parallel.  Likewise, in a chordless cycle of
           length greater than 1, there can be no nodes with self-loops.

        2. We interpret directed two-cycles to be chordless cycles, except in
           multi-digraphs when any edge in a two-cycle has a parallel copy.

        3. We interpret parallel pairs of undirected edges as two-cycles, except
           when a third (or more) parallel edge exists between the two nodes.

        4. Generalizing the above, edges with parallel clones may not occur in
           chordless cycles.

    In a directed graph, two chordless cycles are distinct if they are not
    cyclic permutations of each other.  In an undirected graph, two chordless
    cycles are distinct if they are not cyclic permutations of each other nor of
    the other's reversal.

    Optionally, the cycles are bounded in length.

    We use an algorithm strongly inspired by that of Dias et al [1]_.  It has
    been modified in the following ways:

        1. Recursion is avoided, per Python's limitations

        2. The labeling function is not necessary, because the starting paths
            are chosen (and deleted from the host graph) to prevent multiple
            occurrences of the same path

        3. The search is optionally bounded at a specified length

        4. Support for directed graphs is provided by extending cycles along
            forward edges, and blocking nodes along forward and reverse edges

        5. Support for multigraphs is provided by omitting digons from the set
            of forward edges

    Parameters
    ----------
    G : NetworkX DiGraph
       A directed graph

    length_bound : int or None, optional (default=None)
       If length_bound is an int, generate all simple cycles of G with length at
       most length_bound.  Otherwise, generate all simple cycles of G.

    Yields
    ------
    list of nodes
       Each cycle is represented by a list of nodes along the cycle.

    Examples
    --------
    >>> sorted(list(nx.chordless_cycles(nx.complete_graph(4))))
    [[1, 0, 2], [1, 0, 3], [2, 0, 3], [2, 1, 3]]

    Notes
    -----
    When length_bound is None, and the graph is simple, the time complexity is
    $O((n+e)(c+1))$ for $n$ nodes, $e$ edges and $c$ chordless cycles.

    Raises
    ------
    ValueError
        when length_bound < 0.

    References
    ----------
    .. [1] Efficient enumeration of chordless cycles
       E. Dias and D. Castonguay and H. Longo and W.A.R. Jradi
       https://arxiv.org/abs/1309.1051

    See Also
    --------
    simple_cycles
    """
    if length_bound is not None:
        if length_bound == 0:
            return
        elif length_bound < 0:
            raise ValueError('length bound must be non-negative')
    directed = G.is_directed()
    multigraph = G.is_multigraph()
    if multigraph:
        yield from ([v] for v, Gv in G.adj.items() if len(Gv.get(v, ())) == 1)
    else:
        yield from ([v] for v, Gv in G.adj.items() if v in Gv)
    if length_bound is not None and length_bound == 1:
        return
    if directed:
        F = nx.DiGraph(((u, v) for u, Gu in G.adj.items() if u not in Gu for v in Gu))
        B = F.to_undirected(as_view=False)
    else:
        F = nx.Graph(((u, v) for u, Gu in G.adj.items() if u not in Gu for v in Gu))
        B = None
    if multigraph:
        if not directed:
            B = F.copy()
            visited = set()
        for u, Gu in G.adj.items():
            if directed:
                multiplicity = ((v, len(Guv)) for v, Guv in Gu.items())
                for v, m in multiplicity:
                    if m > 1:
                        F.remove_edges_from(((u, v), (v, u)))
            else:
                multiplicity = ((v, len(Guv)) for v, Guv in Gu.items() if v in visited)
                for v, m in multiplicity:
                    if m == 2:
                        yield [u, v]
                    if m > 1:
                        F.remove_edge(u, v)
                visited.add(u)
    if directed:
        for u, Fu in F.adj.items():
            digons = [[u, v] for v in Fu if F.has_edge(v, u)]
            yield from digons
            F.remove_edges_from(digons)
            F.remove_edges_from((e[::-1] for e in digons))
    if length_bound is not None and length_bound == 2:
        return
    if directed:
        separate = nx.strongly_connected_components

        def stems(C, v):
            for u, w in product(C.pred[v], C.succ[v]):
                if not G.has_edge(u, w):
                    yield ([u, v, w], F.has_edge(w, u))
    else:
        separate = nx.biconnected_components

        def stems(C, v):
            yield from (([u, v, w], F.has_edge(w, u)) for u, w in combinations(C[v], 2))
    components = [c for c in separate(F) if len(c) > 2]
    while components:
        c = components.pop()
        v = next(iter(c))
        Fc = F.subgraph(c)
        Fcc = Bcc = None
        for S, is_triangle in stems(Fc, v):
            if is_triangle:
                yield S
            else:
                if Fcc is None:
                    Fcc = _NeighborhoodCache(Fc)
                    Bcc = Fcc if B is None else _NeighborhoodCache(B.subgraph(c))
                yield from _chordless_cycle_search(Fcc, Bcc, S, length_bound)
        components.extend((c for c in separate(F.subgraph(c - {v})) if len(c) > 2))