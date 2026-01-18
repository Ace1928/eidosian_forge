from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base
def graph_certificate(gr):
    """
    Return a certificate for the graph

    Parameters
    ==========

    gr : adjacency list

    Explanation
    ===========

    The graph is assumed to be unoriented and without
    external lines.

    Associate to each vertex of the graph a symmetric tensor with
    number of indices equal to the degree of the vertex; indices
    are contracted when they correspond to the same line of the graph.
    The canonical form of the tensor gives a certificate for the graph.

    This is not an efficient algorithm to get the certificate of a graph.

    Examples
    ========

    >>> from sympy.combinatorics.testutil import graph_certificate
    >>> gr1 = {0:[1, 2, 3, 5], 1:[0, 2, 4], 2:[0, 1, 3, 4], 3:[0, 2, 4], 4:[1, 2, 3, 5], 5:[0, 4]}
    >>> gr2 = {0:[1, 5], 1:[0, 2, 3, 4], 2:[1, 3, 5], 3:[1, 2, 4, 5], 4:[1, 3, 5], 5:[0, 2, 3, 4]}
    >>> c1 = graph_certificate(gr1)
    >>> c2 = graph_certificate(gr2)
    >>> c1
    [0, 2, 4, 6, 1, 8, 10, 12, 3, 14, 16, 18, 5, 9, 15, 7, 11, 17, 13, 19, 20, 21]
    >>> c1 == c2
    True
    """
    from sympy.combinatorics.permutations import _af_invert
    from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, canonicalize
    items = list(gr.items())
    items.sort(key=lambda x: len(x[1]), reverse=True)
    pvert = [x[0] for x in items]
    pvert = _af_invert(pvert)
    num_indices = 0
    for v, neigh in items:
        num_indices += len(neigh)
    vertices = [[] for i in items]
    i = 0
    for v, neigh in items:
        for v2 in neigh:
            if pvert[v] < pvert[v2]:
                vertices[pvert[v]].append(i)
                vertices[pvert[v2]].append(i + 1)
                i += 2
    g = []
    for v in vertices:
        g.extend(v)
    assert len(g) == num_indices
    g += [num_indices, num_indices + 1]
    size = num_indices + 2
    assert sorted(g) == list(range(size))
    g = Permutation(g)
    vlen = [0] * (len(vertices[0]) + 1)
    for neigh in vertices:
        vlen[len(neigh)] += 1
    v = []
    for i in range(len(vlen)):
        n = vlen[i]
        if n:
            base, gens = get_symmetric_group_sgs(i)
            v.append((base, gens, n, 0))
    v.reverse()
    dummies = list(range(num_indices))
    can = canonicalize(g, dummies, 0, *v)
    return can